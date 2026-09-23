#!/usr/bin/env python3
"""
JEV CHESS — веб-шахматы против Jev.

Режимы:
  * человек против Jev;
  * Stockfish против Jev (смотрим со стороны).

Кто что делает:
  Jev         — выбирает ход: либо сразу, одним `choice` среди всех легальных ходов,
                либо после обдумывания дерева вариантов (jevsearch.py). Решает всегда Jev.
  python-chess — правила: легальность, шах, мат, пат, ничьи. Сервер — единственный
                 источник истины, клиент лишь присылает список ходов в UCI.
  Stockfish   — соперник в режиме «движок против Jev».

Запуск:  .venv/bin/python server.py   →   http://localhost:8766
"""

import http.server
import json
import os
import shutil
import socketserver
import sys
import threading
import time

import chess
import chess.engine

from jev import Jev, JevError, JevRateLimited
from jevdeep import DeepThinker, Settings
from jevsearch import JevThinker, base_state, policy_question

HERE = os.path.dirname(os.path.abspath(__file__))
PORT = int(os.environ.get("PORT", 8766))

def load_env() -> None:
    path = os.path.join(HERE, ".env")
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


# --------------------------------------------------------------------------
# Правила
# --------------------------------------------------------------------------

class BadMove(ValueError):
    pass


def replay(moves: list[str]) -> chess.Board:
    """Собрать позицию из списка UCI-ходов, проверяя легальность каждого."""
    board = chess.Board()
    for i, uci in enumerate(moves):
        try:
            move = chess.Move.from_uci(uci)
        except ValueError:
            raise BadMove(f"ход #{i + 1} «{uci}» — не UCI") from None
        if move not in board.legal_moves:
            raise BadMove(f"ход #{i + 1} «{uci}» нелегален в позиции {board.fen()}")
        board.push(move)
    return board


def status(board: chess.Board) -> dict:
    outcome = board.outcome(claim_draw=True)
    if outcome is None:
        return {"over": False, "check": board.is_check(), "text": ""}
    reasons = {
        chess.Termination.CHECKMATE: "мат",
        chess.Termination.STALEMATE: "пат",
        chess.Termination.INSUFFICIENT_MATERIAL: "недостаточно материала",
        chess.Termination.SEVENTYFIVE_MOVES: "правило 75 ходов",
        chess.Termination.FIVEFOLD_REPETITION: "пятикратное повторение",
        chess.Termination.FIFTY_MOVES: "правило 50 ходов",
        chess.Termination.THREEFOLD_REPETITION: "троекратное повторение",
    }
    winner = {True: "white", False: "black", None: None}[outcome.winner]
    return {
        "over": True,
        "check": board.is_check(),
        "winner": winner,
        "reason": reasons.get(outcome.termination, outcome.termination.name.lower()),
        "result": outcome.result(),
    }


def snapshot(board: chess.Board) -> dict:
    """Всё, что нужно клиенту, чтобы нарисовать позицию и принимать ходы."""
    san, tmp = [], chess.Board()
    for m in board.move_stack:
        san.append(tmp.san(m))
        tmp.push(m)
    return {
        "fen": board.fen(),
        "turn": "white" if board.turn else "black",
        "legal": [m.uci() for m in board.legal_moves],
        "history": [m.uci() for m in board.move_stack],
        "san": san,
        "last": board.move_stack[-1].uci() if board.move_stack else None,
        "status": status(board),
    }


# --------------------------------------------------------------------------
# Игроки
# --------------------------------------------------------------------------

class JevPlayer:
    """Три уровня:
      intuition — один запрос: Jev смотрит на позицию и сразу выбирает ход;
      think     — Jev обдумывает дерево вариантов, MCTS (jevsearch.py);
      deep      — выборочный минимакс с досчётом разменов (jevdeep.py);
      deep_max  — то же, но каждая оценка усредняется с зеркальной доской (×2 запросов).
    Во всех случаях ход выбирает Jev.
    """

    def __init__(self):
        self.jev = Jev(timeout=30, retries=4, rate_limit_wait=45)
        self.thinker = JevThinker(self.jev)
        self.deep = DeepThinker(self.jev)
        self.deep_max = DeepThinker(self.jev, Settings(mirror=True))

    def move(self, board: chess.Board, style: str = "think", budget: int = 48) -> dict:
        legal = list(board.legal_moves)
        if len(legal) == 1:
            m = legal[0]
            return {"uci": m.uci(), "san": board.san(m), "top": [[board.san(m), 1.0]], "forced": True}

        if style in ("deep", "deep_max"):
            thinker = self.deep_max if style == "deep_max" else self.deep
            return {**thinker.think(board), "style": style}

        if style == "think":
            t0 = time.perf_counter()
            r = self.thinker.think(board, budget=max(8, min(200, int(budget))))
            return {**r, "style": "think", "latency": round(time.perf_counter() - t0, 2)}

        a = self.jev.ask(base_state(board), move=policy_question(board, legal))
        pick = a.move
        by_san = {board.san(m): m for m in legal}
        # Защита от невозможного: если ключ вдруг не из списка — берём лучший легальный.
        ranked = [s for s, _ in pick.top(len(legal)) if s in by_san]
        san = pick.value if pick.value in by_san else (ranked[0] if ranked else next(iter(by_san)))
        m = by_san[san]
        return {
            "uci": m.uci(), "san": san, "style": "intuition",
            "top": [[s, round(p, 3)] for s, p in pick.top(5)],
            "latency": round(a.latency, 2), "cost": a.cost,
        }


class EnginePlayer:
    def __init__(self):
        path = os.environ.get("STOCKFISH") or shutil.which("stockfish")
        if not path:
            raise RuntimeError("Stockfish не найден (brew install stockfish или STOCKFISH=/путь)")
        self.engine = chess.engine.SimpleEngine.popen_uci(path)
        self.lock = threading.Lock()

    def move(self, board: chess.Board, skill: int, movetime: float) -> dict:
        skill = max(0, min(20, int(skill)))
        with self.lock:
            self.engine.configure({"Skill Level": skill})
            r = self.engine.play(board, chess.engine.Limit(time=max(0.05, min(5.0, movetime))),
                                 info=chess.engine.INFO_SCORE)
        score = r.info.get("score")
        ev = None
        if score is not None:
            w = score.white()
            ev = f"#{w.mate()}" if w.is_mate() else f"{w.score() / 100:+.2f}"
        return {"uci": r.move.uci(), "san": board.san(r.move), "eval": ev}

    def close(self):
        try:
            self.engine.quit()
        except Exception:
            pass


# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------

JEV: JevPlayer | None = None
ENGINE: EnginePlayer | None = None
ENGINE_ERR = ""


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def send_json(self, code: int, data: dict) -> None:
        body = json.dumps(data, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            with open(os.path.join(HERE, "index.html"), "rb") as f:
                body = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/api/info":
            self.send_json(200, {"engine": ENGINE is not None, "engine_error": ENGINE_ERR})
        else:
            self.send_error(404)

    def do_POST(self):
        try:
            n = int(self.headers.get("Content-Length") or 0)
            req = json.loads(self.rfile.read(n) or b"{}")
            moves = [str(m) for m in req.get("moves", [])]
            board = replay(moves)

            if self.path == "/api/state":
                return self.send_json(200, snapshot(board))

            if self.path == "/api/move":  # ход человека
                uci = str(req.get("move", ""))
                board = replay(moves + [uci])
                return self.send_json(200, snapshot(board))

            if board.is_game_over(claim_draw=True):
                raise BadMove("партия уже окончена")

            if self.path == "/api/jev":
                info = JEV.move(board, str(req.get("style", "deep")), req.get("budget", 48))
                who = "jev"
            elif self.path == "/api/engine":
                if ENGINE is None:
                    return self.send_json(503, {"error": ENGINE_ERR or "движок недоступен"})
                info = ENGINE.move(board, req.get("skill", 5), float(req.get("movetime", 0.3)))
                who = "engine"
            else:
                return self.send_error(404)

            board.push(chess.Move.from_uci(info["uci"]))  # ход уже из legal_moves
            self.send_json(200, {**snapshot(board), "by": who, "info": info})

        except BadMove as e:
            self.send_json(400, {"error": str(e)})
        except JevRateLimited as e:
            self.send_json(429, {"error": "Jev упёрся в лимит OpenRouter", "retry_after": e.retry_after or 5})
        except JevError as e:
            self.send_json(502, {"error": f"Jev: {e}"})
        except Exception as e:  # сеть, таймауты, всё прочее
            self.send_json(500, {"error": f"{type(e).__name__}: {e}"})


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def main() -> None:
    global JEV, ENGINE, ENGINE_ERR
    load_env()
    if not (os.environ.get("OPENROUTER_API_KEY") or os.environ.get("JEV_API_KEY")):
        sys.exit("нужен OPENROUTER_API_KEY (в окружении или в .env)")
    JEV = JevPlayer()
    try:
        ENGINE = EnginePlayer()
    except Exception as e:
        ENGINE_ERR = str(e)
        print(f"  ! {e} — режим «движок против Jev» отключён")

    print(f"\n  JEV CHESS  →  http://localhost:{PORT}\n  Ctrl-C чтобы выйти.\n")
    with Server(("127.0.0.1", PORT), Handler) as srv:
        try:
            srv.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            if ENGINE:
                ENGINE.close()


if __name__ == "__main__":
    main()
