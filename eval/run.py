#!/usr/bin/env python3
"""Прогнать вариант Jev по тестовым позициям и посчитать качество ходов.

Метрики (по оценкам Stockfish из positions.json, Jev их не видит):
  потери  — сколько сантипешек в среднем теряет ход относительно лучшего (обрезано на 1000);
  зевки   — доля ходов, теряющих ≥ 200 сп (фигура или больше);
  лучший  — доля ходов в пределах 10 сп от лучшего.

    .venv/bin/python eval/run.py intuition think48 --limit 150 --parallel 8
"""
import argparse, json, os, sys, time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import chess
import server
from variants import VARIANTS

ap = argparse.ArgumentParser()
ap.add_argument("variants", nargs="+")
ap.add_argument("--limit", type=int, default=150)
ap.add_argument("--parallel", type=int, default=8)
ap.add_argument("--positions", default=os.path.join(HERE, "positions.json"))
a = ap.parse_args()
server.load_env()
positions = json.load(open(a.positions))[: a.limit]

for name in a.variants:
    make = VARIANTS[name]
    player = make()
    t0 = time.time()

    def one(p):
        b = chess.Board()
        for u in p["moves"]:
            b.push(chess.Move.from_uci(u))
        s = time.perf_counter()
        for attempt in range(6):
            try:
                r = player(b)
                break
            except server.JevRateLimited as e:
                time.sleep(e.retry_after or 5)
        else:
            return None
        loss = min(1000, p["best"] - p["evals"][r["uci"]])
        extra = {}
        if r.get("search_best"):
            extra["search_loss"] = min(1000, p["best"] - p["evals"][r["search_best"]])
        return {**extra, "fen": p["fen"], "move": r["uci"], "loss": loss, "cost": r.get("cost", 0) or 0,
                "calls": r.get("calls", 1), "time": time.perf_counter() - s, **{k: r[k] for k in ("override",) if k in r}}

    with ThreadPoolExecutor(a.parallel) as ex:
        res = [r for r in ex.map(one, positions) if r]
    n = len(res)
    out = {
        "variant": name, "n": n,
        "loss": sum(r["loss"] for r in res) / n,
        "blunders": sum(r["loss"] >= 200 for r in res) / n,
        "best": sum(r["loss"] <= 10 for r in res) / n,
        "cost": sum(r["cost"] for r in res) / n,
        "calls": sum(r["calls"] for r in res) / n,
        "time": sum(r["time"] for r in res) / n,
        "wall": time.time() - t0,
    }
    if any("search_loss" in r for r in res):
        sl = [r["search_loss"] for r in res if "search_loss" in r]
        out["search_loss"] = sum(sl) / len(sl)
        out["search_blunders"] = sum(x >= 200 for x in sl) / len(sl)
    if any("override" in r for r in res):
        out["override"] = sum(r.get("override", 0) for r in res) / n
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    json.dump({"summary": out, "rows": res}, open(os.path.join(HERE, "results", f"{name}.json"), "w"), ensure_ascii=False)
    print(f"{name:<22} n={n:<4} потери {out['loss']:6.1f} сп  зевки {out['blunders']*100:5.1f}%  "
          f"лучший {out['best']*100:5.1f}%  ${out['cost']:.4f}/ход  {out['calls']:.0f} запр/ход  "
          f"{out['time']:.1f} с/ход" + (f"  | по расчёту: потери {out['search_loss']:.1f}, зевки {out['search_blunders']*100:.1f}%"
                                         if "search_loss" in out else "")
          + (f"  | Jev отступил от расчёта в {out['override']*100:.0f}%" if "override" in out else ""),
          flush=True)
