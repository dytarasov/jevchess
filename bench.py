#!/usr/bin/env python3
"""
bench.py — матч Jev против Stockfish (или Jev против Jev) без браузера.

    .venv/bin/python bench.py --jev think --budget 48 --vs sf0 --games 4
    .venv/bin/python bench.py --jev think --vs intuition --games 2

Цвета чередуются. В конце печатается строка итогов в формате Markdown-таблицы.
"""

import argparse
import time
from concurrent.futures import ThreadPoolExecutor

import chess
import chess.engine

import server


def play(jev_style: str, budget: int, opponent: str, jev_white: bool, movetime: float) -> dict:
    jp = server.JevPlayer()
    eng = None
    stats = {"cost": 0.0, "jev_moves": 0, "jev_time": 0.0}

    def move(kind: str, board: chess.Board) -> chess.Move:
        nonlocal eng
        if kind.startswith("sf"):
            if eng is None:
                eng = chess.engine.SimpleEngine.popen_uci(server.shutil.which("stockfish"))
                eng.configure({"Skill Level": int(kind[2:])})
            return eng.play(board, chess.engine.Limit(time=movetime)).move
        for _ in range(5):
            try:
                t0 = time.perf_counter()
                r = jp.move(board, kind, budget)
                stats["jev_time"] += time.perf_counter() - t0
                stats["jev_moves"] += 1
                stats["cost"] += r.get("cost") or 0
                return chess.Move.from_uci(r["uci"])
            except server.JevRateLimited as e:
                time.sleep(e.retry_after or 5)
        raise RuntimeError("лимит OpenRouter не отпустил")

    me = jev_style
    white, black = (me, opponent) if jev_white else (opponent, me)
    board = chess.Board()
    try:
        while not board.is_game_over(claim_draw=True) and len(board.move_stack) < 200:
            board.push(move(white if board.turn else black, board))
        out = board.outcome(claim_draw=True)
        if out is None:
            score = 0.5
        else:
            score = 0.5 if out.winner is None else float(out.winner == jev_white)
        result = out.result() if out else "1/2 (200 полуходов)"
        reason = out.termination.name.lower() if out else "limit"
    finally:
        if eng:
            eng.quit()
    return {"white": white, "black": black, "result": result, "reason": reason, "score": score,
            "plies": len(board.move_stack), "rate_limited": jp.jev.rate_limited, **stats}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jev", default="think", choices=["deep", "deep_max", "think", "intuition"])
    ap.add_argument("--budget", type=int, default=48, help="позиций на обдумывание хода")
    ap.add_argument("--vs", default="sf0", help="sf0..sf20 (уровень Stockfish) или intuition/think")
    ap.add_argument("--games", type=int, default=4)
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--movetime", type=float, default=0.05, help="время Stockfish на ход, с")
    a = ap.parse_args()
    server.load_env()

    jobs = [i % 2 == 0 for i in range(a.games)]
    with ThreadPoolExecutor(a.parallel) as ex:
        games = list(ex.map(lambda w: play(a.jev, a.budget, a.vs, w, a.movetime), jobs))

    for g in games:
        print(f"{g['white']:>10} — {g['black']:<10} {g['result']:<8} {g['reason']:<22} "
              f"{g['plies']:>3} полуходов, 429×{g['rate_limited']}")
    wins = sum(g["score"] == 1 for g in games)
    draws = sum(g["score"] == 0.5 for g in games)
    losses = sum(g["score"] == 0 for g in games)
    moves = sum(g["jev_moves"] for g in games) or 1
    label = f"{a.jev} ({a.budget})" if a.jev == "think" else a.jev
    print(f"\n| Jev | соперник | партий | +/=/− | очки | с/ход | $/ход |")
    print(f"|---|---|---|---|---|---|---|")
    print(f"| {label} | {a.vs} | {len(games)} | {wins}/{draws}/{losses} | "
          f"{wins + draws / 2:g}/{len(games)} | {sum(g['jev_time'] for g in games) / moves:.1f} | "
          f"{sum(g['cost'] for g in games) / moves:.4f} |")


if __name__ == "__main__":
    main()
