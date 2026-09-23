#!/usr/bin/env python3
"""Собрать сырые ответы Jev на расширенный набор оценочных вопросов по тестовым позициям
(обычная доска и зеркальная), чтобы сравнивать способы сложить их в оценку без новых запросов."""
import json, os, sys, chess
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
import server; server.load_env()
from concurrent.futures import ThreadPoolExecutor
from jev import Jev
from jevsearch import base_state, ADVANTAGE_Q, MATERIAL_Q, PIECE_RU
from jev import noul

def questions(b):
    qs = {"advantage": ADVANTAGE_Q, "material": MATERIAL_Q,
          "win": noul("Если дальше обе стороны играют сильно, выиграет ли эту партию сторона, которая сейчас ходит?"),
          "tactic": noul("Может ли сторона, которая сейчас ходит, прямо этим ходом или форсированно выиграть материал или поставить мат?"),
          "danger": noul("Есть ли у соперника (стороны, которая НЕ ходит) угроза выиграть материал или поставить мат, если сторона на ходу её проигнорирует?")}
    for sq, p in b.piece_map().items():
        if p.piece_type == chess.KING: continue
        owner = "белых" if p.color else "чёрных"
        qs[f"h_{chess.square_name(sq)}"] = noul(
            f"{PIECE_RU[p.piece_type].capitalize()} {owner} на {chess.square_name(sq)}: может ли соперник выгодно её забрать "
            "(она атакована и не защищена или защищена недостаточно)?")
    return qs

def probe(b):
    a = Jev(retries=4).ask(base_state(b), questions(b))
    raw = {}
    for k, v in a.items():
        raw[k] = {"p": v.p} if hasattr(v, "p") and not hasattr(v, "legend") else {"probs": v.probabilities, "value": v.value}
    return raw

P = json.load(open(os.path.join(HERE, "positions.json")))
boards = []
for p in P:
    b = chess.Board(); [b.push(chess.Move.from_uci(u)) for u in p["moves"]]; boards.append(b)
with ThreadPoolExecutor(12) as ex:
    normal = list(ex.map(probe, boards))
    mirror = list(ex.map(lambda b: probe(b.mirror()), boards))
json.dump({"normal": normal, "mirror": mirror}, open(os.path.join(HERE, "results", "value_probe.json"), "w"))
print("ok", len(normal))
