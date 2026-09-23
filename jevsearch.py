"""
jevsearch.py — «обдумывание» хода, в котором все решения принимает Jev.

Схема похожа на AlphaZero, только вместо нейросети — Jev:

  * В каждой исследуемой позиции Jev одним запросом отвечает на три вопроса:
      policy    — какой ход он бы сыграл (choice среди всех легальных ходов);
      advantage — у кого перевес (score, 7 уровней);
      material  — у кого больше материала (score, 9 уровней).
    Из двух последних складывается оценка позиции.
  * Дерево растёт как в MCTS/PUCT: куда смотреть глубже, определяют вероятности
    и оценки самого Jev. Позиции расширяются волнами, запросы внутри волны
    идут параллельно (латентность Jev почти не зависит от нагрузки).
  * Код не оценивает позиции и не подсказывает ходы. Он только применяет
    правила: двигает фигуры и фиксирует мат, пат и ничьи.
  * Финальный ход выбирает Jev: ему показывают позицию и его же заметки —
    просмотренные линии и его оценки в их концах, в порядке ходов на доске,
    без рейтингов и без «лучшего варианта». Выбрать он может любой легальный ход.
"""

from __future__ import annotations

import math
import threading
from concurrent.futures import ThreadPoolExecutor

import chess

from jev import Jev, choice, score

PIECE_RU = {
    chess.PAWN: "пешка", chess.KNIGHT: "конь", chess.BISHOP: "слон",
    chess.ROOK: "ладья", chess.QUEEN: "ферзь", chess.KING: "король",
}

# Оценка позиции — два вопроса к Jev в том же запросе, с точки зрения белых.
# Замер на 60 позициях против Stockfish: связка «перевес + материал» коррелирует
# с оценкой движка заметно лучше, чем один общий вопрос.
ADVANTAGE_LEVELS = [
    "у чёрных решающий перевес", "у чёрных заметный перевес", "у чёрных небольшой перевес",
    "равно",
    "у белых небольшой перевес", "у белых заметный перевес", "у белых решающий перевес",
]
MATERIAL_LEVELS = [
    "у чёрных больше материала на ферзя или больше", "у чёрных больше на ладью",
    "у чёрных больше на лёгкую фигуру", "у чёрных больше на пешку",
    "материал равный",
    "у белых больше на пешку", "у белых больше на лёгкую фигуру",
    "у белых больше на ладью", "у белых больше на ферзя или больше",
]
VALUE_LABELS = ["проигрываю", "заметно хуже", "немного хуже", "равно",
                "немного лучше", "заметно лучше", "выигрываю"]

MAX_CHILDREN = 8    # сколько ходов-кандидатов раскрывать в узле (по вероятностям Jev)
C_PUCT = 1.6


def describe(board: chess.Board, move: chess.Move) -> str:
    """Нотация хода словами. Только факты правил: что ходит, что берёт, шах/мат."""
    piece = board.piece_at(move.from_square)
    parts = [f"{PIECE_RU[piece.piece_type]} {chess.square_name(move.from_square)}"
             f"→{chess.square_name(move.to_square)}"]
    if board.is_castling(move):
        parts = ["рокировка " + ("короткая" if chess.square_file(move.to_square) == 6 else "длинная")]
    if board.is_capture(move):
        victim = board.piece_at(move.to_square)
        parts.append("берёт " + (PIECE_RU[victim.piece_type] if victim else "пешку на проходе"))
    if move.promotion:
        parts.append("превращение в " + PIECE_RU[move.promotion])
    board.push(move)
    if board.is_checkmate():
        parts.append("МАТ")
    elif board.is_check():
        parts.append("шах")
    board.pop()
    return f"{board.san(move)}: " + ", ".join(parts)


def pgn_text(board: chess.Board) -> str:
    out, tmp = [], chess.Board()
    for i, m in enumerate(board.move_stack):
        if i % 2 == 0:
            out.append(f"{i // 2 + 1}.")
        out.append(tmp.san(m))
        tmp.push(m)
    return " ".join(out) or "(начальная позиция)"


def base_state(board: chess.Board) -> dict:
    side = "белые" if board.turn else "чёрные"
    return {
        "игра": "шахматы",
        "на_ходу": side,
        "fen": board.fen(),
        "доска (заглавные — белые, строчные — чёрные, сверху 8-я горизонталь)": str(board),
        "партия": pgn_text(board),
        "шах_стороне_на_ходу": "да" if board.is_check() else "нет",
        # та же доска, но списком — так Jev заметно точнее видит материал
        "белые_фигуры": _pieces(board, chess.WHITE),
        "чёрные_фигуры": _pieces(board, chess.BLACK),
    }


def _pieces(board: chess.Board, color: bool) -> str:
    items = sorted(board.piece_map().items(), key=lambda kv: (-kv[1].piece_type, kv[0]))
    return ", ".join(f"{PIECE_RU[p.piece_type]} {chess.square_name(sq)}" for sq, p in items if p.color == color)


def policy_question(board: chess.Board, legal: list[chess.Move]) -> dict:
    side = "белыми" if board.turn else "чёрными"
    return choice(
        f"Ты сильный шахматист и играешь {side}. Выбери лучший ход в этой позиции: "
        "не зевай фигуры, ставь мат, если он есть, забирай незащищённый материал, "
        "развивай фигуры и береги короля.",
        {board.san(m): describe(board, m) for m in legal},
    )


ADVANTAGE_Q = score(
    "Оцени позицию: у кого перевес с учётом материала, угроз и безопасности королей?",
    ADVANTAGE_LEVELS,
)
MATERIAL_Q = score(
    "Посчитай материал на доске (пешка 1, конь/слон 3, ладья 5, ферзь 9). У кого больше и насколько?",
    MATERIAL_LEVELS,
)


def value_from(a, board: chess.Board) -> float:
    """Ответы Jev → число в [−1, 1] с точки зрения стороны на ходу."""
    adv = (a.advantage.value - 3) / 3
    mat = (a.material.value - 4) / 4
    v = 0.6 * mat + 0.4 * adv
    return v if board.turn == chess.WHITE else -v


def terminal_value(board: chess.Board) -> float | None:
    """Правила, а не оценка: мат = проигрыш стороны на ходу, ничья = 0."""
    if board.is_checkmate():
        return -1.0
    if (board.is_stalemate() or board.is_insufficient_material()
            or board.can_claim_draw() or board.is_seventyfive_moves()):
        return 0.0
    return None


class Node:
    __slots__ = ("board", "move", "parent", "prior", "N", "W", "children",
                 "expanded", "terminal", "jev_value", "vloss")

    def __init__(self, board: chess.Board, move: chess.Move | None, parent: "Node | None", prior: float):
        self.board = board
        self.move = move
        self.parent = parent
        self.prior = prior
        self.N = 0
        self.W = 0.0          # сумма оценок с точки зрения стороны на ходу в ЭТОМ узле
        self.children: list[Node] = []
        self.expanded = False
        self.terminal = terminal_value(board)
        self.jev_value: float | None = None
        self.vloss = 0

    def q_for_parent(self) -> float:
        n = self.N + self.vloss
        if n == 0:
            return 0.0
        # оценка ребёнка — со стороны соперника; виртуальный проигрыш считаем как +1 сопернику
        return -(self.W + self.vloss) / n


class JevThinker:
    def __init__(self, jev: Jev, concurrency: int = 12):
        self.jev = jev
        self.pool = ThreadPoolExecutor(max_workers=concurrency)
        self.lock = threading.Lock()

    # ---------- один запрос: policy + value ----------

    def evaluate(self, board: chess.Board) -> tuple[dict[chess.Move, float], float, float]:
        legal = list(board.legal_moves)
        if len(legal) == 1:  # единственный ход — спрашивать нечего, нужна только оценка
            a = self.jev.ask(base_state(board), advantage=ADVANTAGE_Q, material=MATERIAL_Q)
            return {legal[0]: 1.0}, value_from(a, board), a.cost
        a = self.jev.ask(base_state(board), policy=policy_question(board, legal),
                         advantage=ADVANTAGE_Q, material=MATERIAL_Q)
        by_san = {board.san(m): m for m in legal}
        priors = {by_san[s]: p for s, p in a.policy.probabilities.items() if s in by_san}
        if not priors:
            priors = {m: 1 / len(legal) for m in legal}
        return priors, value_from(a, board), a.cost

    def expand(self, node: Node, priors: dict[chess.Move, float], v: float) -> None:
        top = sorted(priors.items(), key=lambda kv: -kv[1])[:MAX_CHILDREN]
        total = sum(p for _, p in top) or 1.0
        for m, p in top:
            b = node.board.copy(stack=True)
            b.push(m)
            node.children.append(Node(b, m, node, p / total))
        node.jev_value = v
        node.expanded = True

    # ---------- MCTS-подобный поиск ----------

    def select_leaf(self, root: Node) -> list[Node]:
        path = [root]
        node = root
        while node.expanded and node.terminal is None and node.children:
            sqrt_n = math.sqrt(node.N + node.vloss + 1)
            node = max(node.children, key=lambda c: c.q_for_parent()
                       + C_PUCT * c.prior * sqrt_n / (1 + c.N + c.vloss))
            path.append(node)
        return path

    @staticmethod
    def backup(path: list[Node], v: float) -> None:
        for node in reversed(path):
            node.N += 1
            node.W += v
            v = -v

    def think(self, board: chess.Board, budget: int = 48, batch: int = 8) -> dict:
        root = Node(board.copy(stack=True), None, None, 1.0)
        priors, v, cost = self.evaluate(root.board)
        self.expand(root, priors, v)
        root.N, root.W = 1, v
        calls, positions = 1, 1

        spins = 0
        while calls < budget and spins < budget * 4:
            spins += 1
            wave: list[list[Node]] = []
            seen = set()
            for _ in range(min(batch, budget - calls)):
                path = self.select_leaf(root)
                leaf = path[-1]
                if leaf.terminal is not None:
                    self.backup(path, leaf.terminal)
                    continue
                if id(leaf) in seen:
                    continue
                seen.add(id(leaf))
                for n in path:
                    n.vloss += 1
                wave.append(path)
            if not wave:
                continue

            futures = [self.pool.submit(self.evaluate, p[-1].board) for p in wave]
            failed = 0
            for path, fut in zip(wave, futures):
                for n in path:
                    n.vloss -= 1
                try:
                    pr, val, c = fut.result()
                except Exception:
                    # лимиты/сеть: эту позицию просто не раскрываем в этой волне
                    failed += 1
                    continue
                self.expand(path[-1], pr, val)
                self.backup(path, val)
                cost += c
                positions += 1
            calls += len(wave)
            if failed == len(wave):
                break  # Jev недоступен — решаем по тому, что успели обдумать

        notes, lines = self.notes(root)
        decision, dcost = self.decide(board, notes)
        return {**decision, "lines": lines, "positions": positions,
                "cost": cost + dcost, "calls": calls + 1}

    # ---------- заметки и финальное решение ----------

    @staticmethod
    def line_of(child: Node) -> tuple[list[str], float]:
        """Главная линия после хода-кандидата: идём по самым посещаемым ответам."""
        sans, node = [], child
        while True:
            parent_board = node.parent.board
            sans.append(parent_board.san(node.move))
            if not node.children or not any(c.N for c in node.children) or len(sans) >= 6:
                break
            node = max(node.children, key=lambda c: c.N)
        # оценка линии с точки зрения корня: средняя оценка поддерева кандидата
        q = -child.W / child.N if child.N else 0.0
        return sans, q

    def notes(self, root: Node) -> tuple[list[str], list[dict]]:
        me_white = root.board.turn
        notes, lines = [], []
        # порядок — как ходы перечислены на доске, а не по «силе», чтобы не подсказывать
        order = {m: i for i, m in enumerate(root.board.legal_moves)}
        for child in sorted(root.children, key=lambda c: order[c.move]):
            if child.N == 0:
                continue
            sans, q = self.line_of(child)
            if child.terminal == -1.0:
                label = "ставлю мат"
            else:
                label = VALUE_LABELS[max(0, min(6, round(q * 3 + 3)))]
            full_move = root.board.fullmove_number
            text = _numbered(sans, full_move, me_white)
            notes.append(f"{text} — моя оценка в конце: {label}")
            lines.append({"san": sans[0], "line": text, "label": label, "visits": child.N, "q": round(q, 3)})
        return notes, lines

    def decide(self, board: chess.Board, notes: list[str]) -> tuple[dict, float]:
        legal = list(board.legal_moves)
        side = "белыми" if board.turn else "чёрными"
        state = base_state(board)
        state["мои_заметки_после_обдумывания"] = "\n".join(notes) or "(ничего не успел обдумать)"
        q = choice(
            f"Ты сильный шахматист и играешь {side}. Перед этим ты обдумал варианты — "
            "это твои собственные заметки: линии, которые ты просчитал, и как ты оценил "
            "позицию в их конце. Учитывая свой расчёт, выбери ход, который сыграешь.",
            {board.san(m): describe(board, m) for m in legal},
        )
        a = self.jev.ask(state, move=q)
        by_san = {board.san(m): m for m in legal}
        ranked = [s for s, _ in a.move.top(len(legal)) if s in by_san]
        san = a.move.value if a.move.value in by_san else (ranked[0] if ranked else next(iter(by_san)))
        m = by_san[san]
        return {"uci": m.uci(), "san": san,
                "top": [[s, round(p, 3)] for s, p in a.move.top(5)]}, a.cost


def _numbered(sans: list[str], fullmove: int, white_first: bool) -> str:
    out, n, white = [], fullmove, white_first
    for i, s in enumerate(sans):
        if white:
            out.append(f"{n}.{s}")
        else:
            out.append(f"{n}...{s}" if i == 0 else s)
            n += 1
        white = not white
    return " ".join(out)
