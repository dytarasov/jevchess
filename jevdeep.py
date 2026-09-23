"""
jevdeep.py — «глубокое» обдумывание: выборочный минимакс, в котором всё решает Jev.

Чем отличается от jevsearch.py (MCTS):
  * Ход оценивается по ЛУЧШЕМУ ответу соперника (минимакс), а не по среднему:
    один опровергающий ответ перевешивает семь безобидных.
  * Размены досчитываются (quiescence): позицию не оценивают посреди размена.
    Какие взятия смотреть — решает Jev (его вероятности), плюс ответное взятие
    на поле последнего хода.
  * Ответы соперника: самые вероятные по мнению Jev плюс все его взятия и шахи.
    Это правило перебора («проверь форсированные ходы»), а не оценка: код не
    знает, хорошие они или нет, — это потом скажет Jev.

Что делает Jev:
  * в каждой позиции — policy (какой ход сыграл бы) и оценку позиции
    (материал + перевес, в пешках, с точки зрения стороны на ходу);
  * финальный выбор хода: он видит свои линии и свои оценки в порядке ходов на
    доске, без рейтинга, и может сыграть любой легальный ход.

Что делает код: правила (легальность, мат, пат, ничьи), обход дерева и минимакс
над оценками Jev.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import chess

from jev import Jev, choice, score
from jevsearch import (ADVANTAGE_LEVELS, base_state, describe,
                       policy_question, _numbered)

MATE = 100.0
ADV_PAWNS = [-6, -3, -1, 0, 1, 3, 6]             # уровни ADVANTAGE_LEVELS в пешках (за белых)

ADVANTAGE_Q = score(
    "Оцени позицию: у кого перевес с учётом материала, угроз и безопасности королей?",
    ADVANTAGE_LEVELS,
)
# Материал Jev считает поштучно: «сколько у белых ладей?» и т.д. Один общий вопрос
# «у кого больше материала» он путал (ошибка ≥ 3 пешек в 52 из 150 позиций),
# а поштучный подсчёт верен в 96% ответов (ошибка ≥ 3 пешек — 1 из 150).
# Код только складывает ответы Jev с общепринятой ценностью фигур.
PIECE_VALUE = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
_PLURAL = {chess.PAWN: "пешек", chess.KNIGHT: "коней", chess.BISHOP: "слонов",
           chess.ROOK: "ладей", chess.QUEEN: "ферзей"}
COUNT_QS = {
    f"{'w' if color else 'b'}{pt}": choice(
        f"Сколько {_PLURAL[pt]} у {'белых' if color else 'чёрных'} сейчас на доске?",
        {str(i): str(i) for i in range(9 if pt == chess.PAWN else 4)},
    )
    for color in (chess.WHITE, chess.BLACK) for pt in PIECE_VALUE
}


def counted_material(a) -> float:
    """Материал за белых в пешках — по поштучным ответам Jev (матожидание)."""
    total = 0.0
    for pt, val in PIECE_VALUE.items():
        w = sum(int(k) * p for k, p in a[f"w{pt}"].probabilities.items())
        b = sum(int(k) * p for k, p in a[f"b{pt}"].probabilities.items())
        total += val * (w - b)
    return total


@dataclass
class Settings:
    widths: tuple[int, ...] = (8, 8)     # сколько ходов Jev смотреть на каждом полуходе
    forcing_plies: tuple[int, ...] = (1,)  # на каких полуходах добавлять все взятия и шахи
    root_mass: float = 0.95              # на корне — пока суммарная вероятность Jev не наберёт столько
    q_depth: int = 3                     # глубина досчёта разменов
    q_width: int = 1                     # сколько взятий (по Jev) смотреть в размене
    forcing_checks: bool = False         # добавлять ли к взятиям соперника ещё и все шахи
    max_positions: int = 240             # общий бюджет позиций на ход
    mirror: bool = False                 # усреднять оценку с зеркальной доской (×2 запроса)
    concurrency: int = 24


@dataclass
class Node:
    board: chess.Board
    move: chess.Move | None = None
    ply: int = 0
    kind: str = "full"                   # full — обязан ходить; q — может «стоять» (stand pat)
    path_p: float = 1.0                  # произведение вероятностей Jev вдоль линии — порядок при нехватке бюджета
    parent: "Node | None" = None
    children: list["Node"] = field(default_factory=list)
    priors: dict = field(default_factory=dict)
    static: float | None = None          # оценка Jev, пешки, со стороны стороны на ходу
    value: float | None = None           # после минимакса
    terminal: float | None = None


def terminal_value(board: chess.Board) -> float | None:
    if board.is_checkmate():
        return -MATE
    if (board.is_stalemate() or board.is_insufficient_material()
            or board.can_claim_draw() or board.is_seventyfive_moves()):
        return 0.0
    return None


def expectation(probs: dict, table: list[int]) -> float:
    return sum(table[int(k)] * float(p) for k, p in probs.items())


class DeepThinker:
    def __init__(self, jev: Jev, settings: Settings | None = None):
        self.jev = jev
        self.s = settings or Settings()
        self.pool = ThreadPoolExecutor(max_workers=self.s.concurrency)

    # ---------- Jev: одна позиция ----------

    def _ask(self, board: chess.Board, with_policy: bool) -> tuple[dict, float, float, int]:
        legal = list(board.legal_moves)
        qs = {"advantage": ADVANTAGE_Q, **COUNT_QS}
        if with_policy and len(legal) > 1:
            qs["policy"] = policy_question(board, legal)
        a = self.jev.ask(base_state(board), qs)
        sign = 1 if board.turn == chess.WHITE else -1
        v = sign * (counted_material(a) + 0.5 * expectation(a.advantage.probabilities, ADV_PAWNS))
        priors = {}
        if "policy" in qs:
            by_san = {board.san(m): m for m in legal}
            priors = {by_san[s]: p for s, p in a.policy.probabilities.items() if s in by_san}
        elif len(legal) == 1:
            priors = {legal[0]: 1.0}
        return priors, v, a.cost, 1

    def evaluate(self, node: Node) -> tuple[float, int]:
        priors, v, cost, calls = self._ask(node.board, True)
        if self.s.mirror:
            _, vm, c2, _ = self._ask(node.board.mirror(), False)
            v, cost, calls = (v + vm) / 2, cost + c2, calls + 1
        node.priors, node.static = priors, v
        return cost, calls

    # ---------- какие ходы смотреть ----------

    def _full_moves(self, node: Node) -> list[chess.Move]:
        b = node.board
        ranked = [m for m, _ in sorted(node.priors.items(), key=lambda kv: -kv[1])]
        if node.ply == 0:
            out, mass = [], 0.0
            for m in ranked:
                out.append(m)
                mass += node.priors[m]
                if len(out) >= self.s.widths[0] or (mass >= self.s.root_mass and len(out) >= 3):
                    break
        else:
            out = ranked[: self.s.widths[node.ply]]
        if node.ply in self.s.forcing_plies:
            for m in b.legal_moves:
                if m not in out and (b.is_capture(m) or (self.s.forcing_checks and b.gives_check(m))):
                    out.append(m)
        return out or list(b.legal_moves)[:1]

    def _q_moves(self, node: Node) -> list[chess.Move]:
        b = node.board
        if b.is_check():  # под шахом стоять нельзя — смотрим ответы по Jev
            ranked = sorted(node.priors.items(), key=lambda kv: -kv[1])
            return [m for m, _ in ranked[:3]] or list(b.legal_moves)[:3]
        caps = sorted(((m, node.priors.get(m, 0.0)) for m in b.legal_moves if b.is_capture(m)),
                      key=lambda kv: -kv[1])
        out = [m for m, _ in caps[: self.s.q_width]]
        last = node.move.to_square if node.move else None
        for m, _ in caps:  # ответное взятие на поле последнего хода
            if m.to_square == last and m not in out:
                out.append(m)
                break
        return out

    # ---------- обход ----------

    def think(self, board: chess.Board) -> dict:
        t0 = time.perf_counter()
        root = Node(board.copy(stack=True))
        cost, calls, positions = 0.0, 0, 0
        cache: dict[str, Node] = {}

        def run_wave(nodes: list[Node]) -> None:
            nonlocal cost, calls, positions
            todo = []
            for n in nodes:
                n.terminal = terminal_value(n.board)
                if n.terminal is not None:
                    continue
                key = n.board.fen()
                if key in cache:
                    n.priors, n.static = cache[key].priors, cache[key].static
                else:
                    cache[key] = n
                    todo.append(n)
            for c, k in self.pool.map(self.evaluate, todo):
                cost += c
                calls += k
            positions += len(todo)

        run_wave([root])
        frontier = [root]
        depth = len(self.s.widths)
        # полные полуходы
        for ply in range(depth):
            nxt = []
            for n in frontier:
                if n.terminal is not None:
                    continue
                for m in self._full_moves(n):
                    b = n.board.copy(stack=True)
                    b.push(m)
                    kind = "full" if ply + 1 < depth else "q"
                    c = Node(b, m, ply + 1, kind, path_p=n.path_p * max(n.priors.get(m, 0.0), 0.02))
                    n.children.append(c)
                    nxt.append(c)
            run_wave(nxt)
            frontier = nxt
        # досчёт разменов
        for _ in range(self.s.q_depth):
            nxt = []
            for n in frontier:
                if n.terminal is not None or n.static is None:
                    continue
                for m in self._q_moves(n):
                    b = n.board.copy(stack=True)
                    b.push(m)
                    nxt.append(Node(b, m, n.ply + 1, "q", parent=n,
                                    path_p=n.path_p * max(n.priors.get(m, 0.0), 0.02)))
            room = self.s.max_positions - positions
            if not nxt or room <= 0:
                break
            # бюджет кончается — сначала линии, которые Jev считает более вероятными
            nxt = sorted(nxt, key=lambda c: -c.path_p)[:room]
            for c in nxt:
                c.parent.children.append(c)
            run_wave(nxt)
            frontier = nxt

        self._negamax(root)
        notes, lines, best_move = self._notes(root)
        decision, dcost = self._decide(board, notes)
        return {**decision, "lines": lines, "positions": positions, "calls": calls + 1,
                "cost": cost + dcost, "search_best": best_move,
                "override": int(best_move is not None and decision["uci"] != best_move),
                "latency": round(time.perf_counter() - t0, 2)}

    def _negamax(self, n: Node) -> float:
        if n.terminal is not None:
            n.value = n.terminal - (0.01 * n.ply if n.terminal < 0 else 0)  # мат ближе — хуже
            return n.value
        if n.static is None:
            n.value = 0.0
            return 0.0
        if not n.children:
            n.value = n.static
            return n.value
        best = -1e9
        if n.kind == "q" and not n.board.is_check():
            best = n.static                    # stand pat: можно не продолжать размен
        for c in n.children:
            best = max(best, -self._negamax(c))
        n.value = best
        return best

    # ---------- заметки и решение Jev ----------

    @staticmethod
    def _label(v: float) -> str:
        if v >= MATE / 2:
            return "ставлю мат"
        if v <= -MATE / 2:
            return "мне ставят мат"
        words = ("выигрываю решающе" if v >= 5 else "заметно лучше" if v >= 2 else
                 "немного лучше" if v >= 0.6 else "примерно равно" if v > -0.6 else
                 "немного хуже" if v > -2 else "заметно хуже" if v > -5 else "проигрываю")
        return f"{words} ({v:+.1f})"

    def _pv(self, n: Node) -> list[str]:
        sans = []
        while n.children:
            nxt = min(n.children, key=lambda c: c.value if c.value is not None else 1e9)
            if n.kind == "q" and n.static is not None and -nxt.value <= n.static and not n.board.is_check():
                break  # здесь выгоднее не продолжать
            sans.append(n.board.san(nxt.move))
            n = nxt
            if len(sans) >= 6:
                break
        return sans

    def _notes(self, root: Node):
        order = {m: i for i, m in enumerate(root.board.legal_moves)}
        notes, lines = [], []
        best, best_v = None, -1e9
        for c in sorted(root.children, key=lambda c: order[c.move]):
            v = -c.value if c.value is not None else 0.0
            if v > best_v:
                best, best_v = c.move.uci(), v
            sans = [root.board.san(c.move)] + self._pv(c)
            text = _numbered(sans, root.board.fullmove_number, root.board.turn)
            label = self._label(v)
            notes.append(f"{text} — при лучшей игре соперника, по моей оценке: {label}")
            lines.append({"san": sans[0], "line": text, "label": label, "q": round(v, 2)})
        return notes, lines, best

    def _decide(self, board: chess.Board, notes: list[str]) -> tuple[dict, float]:
        legal = list(board.legal_moves)
        if len(legal) == 1:
            return {"uci": legal[0].uci(), "san": board.san(legal[0]), "top": [[board.san(legal[0]), 1.0]]}, 0.0
        side = "белыми" if board.turn else "чёрными"
        state = base_state(board)
        state["мои_заметки_после_расчёта"] = "\n".join(notes) or "(ничего не успел просчитать)"
        q = choice(
            f"Ты сильный шахматист и играешь {side}. Перед этим ты просчитал варианты — "
            "это твои собственные заметки: для каждого хода линия с лучшими, по-твоему, "
            "ответами соперника и твоя оценка позиции в её конце. Учитывая свой расчёт, "
            "выбери ход, который сыграешь.",
            {board.san(m): describe(board, m) for m in legal},
        )
        a = self.jev.ask(state, move=q)
        by_san = {board.san(m): m for m in legal}
        san = a.move.value if a.move.value in by_san else next(iter(by_san))
        m = by_san[san]
        return {"uci": m.uci(), "san": san, "top": [[s, round(p, 3)] for s, p in a.move.top(5)]}, a.cost
