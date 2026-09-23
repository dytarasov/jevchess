#!/usr/bin/env python3
"""Разобрать сохранённые партии bench.py: где ходы Jev теряли больше всего (по Stockfish)."""
import glob, json, shutil, sys
import chess, chess.engine
path = sys.argv[1] if len(sys.argv) > 1 else sorted(glob.glob("bench_games/*.json"))[-1]
eng = chess.engine.SimpleEngine.popen_uci(shutil.which("stockfish"))
for gi, g in enumerate(json.load(open(path))):
    b = chess.Board(); worst = []
    for u in g["moves"]:
        m = chess.Move.from_uci(u)
        if b.turn == g["jev_white"]:
            before = eng.analyse(b, chess.engine.Limit(depth=12))["score"].pov(b.turn).score(mate_score=2000)
            san = b.san(m); fen = b.fen(); b.push(m)
            after = -eng.analyse(b, chess.engine.Limit(depth=12))["score"].pov(b.turn).score(mate_score=2000)
            worst.append((before - after, b.fullmove_number, san, fen, before, after))
        else:
            b.push(m)
    worst.sort(reverse=True)
    print(f"партия {gi+1}: Jev {'белыми' if g['jev_white'] else 'чёрными'}, {g['result']} ({g['reason']})")
    for loss, n, san, fen, bef, aft in worst[:4]:
        print(f"   ход {n}: {san:<7} потеря {loss:5d}  ({bef:+d} → {aft:+d})  {fen}")
eng.quit()
