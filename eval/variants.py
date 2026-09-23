"""Реестр вариантов игрока для eval/run.py: имя → фабрика функции board → {uci, cost, calls}."""
import random
import chess
import server


def _intuition():
    jp = server.JevPlayer()
    return lambda b: jp.move(b, "intuition")


def _think(budget):
    def make():
        jp = server.JevPlayer()
        return lambda b: jp.move(b, "think", budget)
    return make


def _random():
    return lambda b: {"uci": random.choice(list(b.legal_moves)).uci()}


VARIANTS = {
    "random": _random,
    "intuition": _intuition,
    "think48": _think(48),
}


def _deep(decide=True, **kw):
    def make():
        from jev import Jev
        from jevdeep import DeepThinker, Settings
        t = DeepThinker(Jev(timeout=30, retries=4, rate_limit_wait=60), Settings(**kw))
        def play(b):
            r = t.think(b)
            if not decide and r.get("search_best"):   # диагностика: ход по минимаксу, без финального выбора Jev
                r = {**r, "uci": r["search_best"]}
            return r
        return play
    return make


VARIANTS.update({
    "deep": _deep(),
    "deep_pick": _deep(decide=False),
    "deep_mirror": _deep(mirror=True),
    "deep3": _deep(widths=(6, 6, 3), forcing_plies=(1,)),
})
