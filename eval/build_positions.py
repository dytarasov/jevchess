#!/usr/bin/env python3
"""Собрать тестовый набор позиций с оценкой Stockfish для КАЖДОГО легального хода.

Позиции берутся из партий слабых движков с примесью случайных ходов — так в набор
попадают и спокойные позиции, и позиции с висящими фигурами и тактикой.
Stockfish здесь — только экзаменатор: Jev эти оценки никогда не видит.

    .venv/bin/python eval/build_positions.py --n 150 --out eval/positions.json
"""
import argparse, json, random, shutil
import chess, chess.engine

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=150)
ap.add_argument("--depth", type=int, default=12)
ap.add_argument("--seed", type=int, default=2026)
ap.add_argument("--out", default="eval/positions.json")
a = ap.parse_args()
random.seed(a.seed)
eng = chess.engine.SimpleEngine.popen_uci(shutil.which("stockfish"))
MATE = 1500

out, seen = [], set()
while len(out) < a.n:
    b = chess.Board()
    target = random.randint(10, 90)
    eng.configure({"Skill Level": random.choice([0, 2, 5])})
    while len(b.move_stack) < target and not b.is_game_over():
        m = random.choice(list(b.legal_moves)) if random.random() < 0.15 else eng.play(b, chess.engine.Limit(time=0.02)).move
        b.push(m)
    if b.is_game_over() or b.legal_moves.count() < 2 or b.board_fen() in seen:
        continue
    eng.configure({"Skill Level": 20})
    infos = eng.analyse(b, chess.engine.Limit(depth=a.depth), multipv=b.legal_moves.count())
    evals = {i["pv"][0].uci(): i["score"].pov(b.turn).score(mate_score=MATE) for i in infos}
    evals = {k: max(-MATE, min(MATE, v)) for k, v in evals.items()}
    if len(evals) != b.legal_moves.count():
        continue
    best = max(evals.values())
    if abs(best) > 900:        # уже решённые позиции мало что говорят
        continue
    seen.add(b.board_fen())
    out.append({"fen": b.fen(), "moves": [m.uci() for m in b.move_stack], "evals": evals, "best": best})
    print(len(out), end=" ", flush=True)
eng.quit()
json.dump(out, open(a.out, "w"))
print("\nсохранено", len(out))
