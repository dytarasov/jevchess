"""
jev.py — тонкий типизированный клиент для TypeSafe Jev (System One decision model).

Jev — это не чат-модель. Вход: произвольное состояние. Выход: типизированные
решения с калиброванными вероятностями. Вызывается через альфа-эндпоинт
OpenRouter `/api/alpha/decisions`, а не через /chat/completions.

Экономика, вокруг которой построена обёртка:
  * вход платный ($0.042/Mtok), выход бесплатный;
  * латентность почти не растёт от числа вопросов (200 вопросов ≈ 1 вопрос);
  => задавать N вопросов к одному state одним запросом почти всегда выгоднее,
     чем N запросов. Клиент поощряет именно такой стиль.

Зависимостей нет — только стандартная библиотека.
"""

from __future__ import annotations

import hashlib
import json
import os
import ssl
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

DEFAULT_URL = "https://openrouter.ai/api/alpha/decisions"
DEFAULT_MODEL = "typesafe/jev-1.13"

__all__ = [
    "Jev", "JevError", "noul", "choice", "score",
    "Noul", "Choice", "Score", "Answers", "UNSURE",
]


class JevError(RuntimeError):
    """Ошибка транспорта или валидации на стороне API."""

    def __init__(self, status: int, body: str):
        self.status = status
        self.body = body
        super().__init__(f"HTTP {status}: {body[:500]}")


class _Unsure:
    """Часовой для случая «модель не уверена настолько, насколько мы просили»."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __bool__(self) -> bool:
        return False

    def __repr__(self) -> str:
        return "UNSURE"


UNSURE = _Unsure()


# --------------------------------------------------------------------------
# Построители вопросов
# --------------------------------------------------------------------------

def noul(instructions: str, true: str = "", false: str = "") -> dict:
    """Булев вопрос. Ответ — вероятность истинности (калиброванная, не 0/1).

    >>> noul("Клиент угрожает уйти?", true="угрожает отменой", false="не угрожает")
    """
    return {
        "type": "noul",
        "instructions": instructions,
        "criteria": {
            "true": true or f"да: {instructions}",
            "false": false or f"нет: {instructions}",
        },
    }


def choice(instructions: str, options: Mapping[str, str] | Sequence[str]) -> dict:
    """Выбор одного варианта из набора. `options` — {id: описание} или список id."""
    if not isinstance(options, Mapping):
        options = {str(o): str(o) for o in options}
    if len(options) < 2:
        raise ValueError("choice требует минимум два варианта")
    return {
        "type": "choice",
        "instructions": instructions,
        "criteria": {str(k): _as_text(v) for k, v in options.items()},
    }


def score(instructions: str, levels: Sequence[str]) -> dict:
    """Порядковая шкала. Ответ — дробная позиция на шкале (матожидание)."""
    levels = [_as_text(v) for v in levels]
    if len(levels) < 2:
        raise ValueError("score требует минимум два уровня шкалы")
    return {"type": "score", "instructions": instructions, "criteria": levels}


def _as_text(v: Any) -> str:
    """API требует строки в criteria — структуры кодируем в JSON."""
    return v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)


# --------------------------------------------------------------------------
# Ответы
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Noul:
    """Булево решение с калиброванной вероятностью."""

    p: float

    @property
    def value(self) -> bool:
        return self.p >= 0.5

    @property
    def confidence(self) -> float:
        """0.0 — полная неопределённость (p=0.5), 1.0 — полная уверенность."""
        return abs(self.p - 0.5) * 2

    def sure(self, threshold: float = 0.9) -> bool | _Unsure:
        """Вернуть bool, только если модель уверена; иначе UNSURE (эскалация)."""
        if self.p >= threshold:
            return True
        if self.p <= 1 - threshold:
            return False
        return UNSURE

    def __bool__(self) -> bool:
        return self.value

    def __float__(self) -> float:
        return self.p


@dataclass(frozen=True)
class Choice:
    """Выбор варианта с полным распределением по вариантам."""

    value: str
    probabilities: dict[str, float] = field(default_factory=dict)
    confidence: float = 0.0

    def sure(self, threshold: float = 0.8) -> str | _Unsure:
        return self.value if self.probabilities.get(self.value, 0.0) >= threshold else UNSURE

    @property
    def p(self) -> float:
        """Вероятность выбранного варианта."""
        return self.probabilities.get(self.value, 0.0)

    def top(self, n: int = 3) -> list[tuple[str, float]]:
        return sorted(self.probabilities.items(), key=lambda kv: -kv[1])[:n]

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class Score:
    """Позиция на порядковой шкале (матожидание) + распределение по уровням."""

    value: float
    legend: dict[str, str] = field(default_factory=dict)
    probabilities: dict[str, float] = field(default_factory=dict)
    confidence: float = 0.0

    @property
    def label(self) -> str:
        """Текст ближайшего уровня шкалы."""
        return self.legend.get(str(round(self.value)), "")

    @property
    def normalized(self) -> float:
        """Значение, нормированное в [0, 1] по длине шкалы."""
        top = max(1, len(self.legend) - 1)
        return self.value / top

    def at_least(self, level: float) -> bool:
        return self.value >= level

    def __float__(self) -> float:
        return self.value


def _parse_answer(raw: Mapping[str, Any]) -> Noul | Choice | Score:
    kind = raw.get("type")
    if kind == "noul":
        return Noul(p=float(raw["noul"]))
    if kind == "choice":
        return Choice(
            value=raw["choice"],
            probabilities={k: float(v) for k, v in (raw.get("probabilities") or {}).items()},
            confidence=float(raw.get("confidence", 0.0)),
        )
    if kind == "score":
        return Score(
            value=float(raw["score"]),
            legend=dict(raw.get("legend") or {}),
            probabilities={k: float(v) for k, v in (raw.get("probabilities") or {}).items()},
            confidence=float(raw.get("confidence", 0.0)),
        )
    raise JevError(0, f"неизвестный тип ответа: {kind!r}")


class Answers(Mapping):
    """Ответы на один запрос. Доступ по ключу и через атрибут."""

    def __init__(self, answers: dict, usage: dict, raw: dict, latency: float):
        self._a = answers
        self.usage = usage
        self.raw = raw
        self.latency = latency

    # Mapping
    def __getitem__(self, k: str): return self._a[k]
    def __iter__(self) -> Iterator[str]: return iter(self._a)
    def __len__(self) -> int: return len(self._a)

    def __getattr__(self, k: str):
        try:
            return self._a[k]
        except KeyError:
            raise AttributeError(k) from None

    @property
    def cost(self) -> float:
        return float(self.usage.get("cost", 0.0))

    def values_dict(self) -> dict[str, Any]:
        """Плоский словарь {ключ: питоновское значение} — удобно для таблиц и CSV."""
        out = {}
        for k, v in self._a.items():
            out[k] = v.p if isinstance(v, Noul) else (v.value if isinstance(v, (Choice, Score)) else v)
        return out

    def __repr__(self) -> str:
        inner = ", ".join(f"{k}={v!r}" for k, v in self._a.items())
        return f"Answers({inner})"


# --------------------------------------------------------------------------
# Клиент
# --------------------------------------------------------------------------

class Jev:
    """Клиент решений.

    >>> jev = Jev()
    >>> a = jev.ask("клиент грозит уйти к конкурентам",
    ...             churn=noul("Есть ли угроза оттока?"))
    >>> bool(a.churn), a.churn.p
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_URL,
        timeout: float = 60.0,
        retries: int = 3,
        cache_dir: str | None = None,
        concurrency: int = 8,
    ):
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY") or os.environ.get("JEV_API_KEY")
        if not self.api_key:
            raise ValueError("нужен API-ключ: аргумент api_key или переменная OPENROUTER_API_KEY")
        self.model = model
        self.base_url = base_url
        self.timeout = timeout
        self.retries = retries
        self.concurrency = concurrency
        self.cache_dir = cache_dir
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

        # Телеметрия — Jev достаточно дёшев, чтобы его звали в цикле; полезно видеть счёт.
        self.calls = 0
        self.total_cost = 0.0
        self.total_input_tokens = 0
        self.cache_hits = 0
        self._lock = threading.Lock()
        self._ssl = _ssl_context()

    # ---------------- основной вызов ----------------

    def ask(
        self,
        state: str | Mapping | Sequence,
        questions: Mapping[str, dict] | None = None,
        **kwargs: dict,
    ) -> Answers:
        """Задать один или несколько вопросов об одном состоянии — одним запросом.

        Вопросы передаются словарём или именованными аргументами:
            jev.ask(text, urgent=noul("Срочно?"), team=choice("Кому?", {...}))
        """
        qs = dict(questions or {})
        qs.update(kwargs)
        if not qs:
            raise ValueError("нужен хотя бы один вопрос")

        payload = {"model": self.model, "state": _prepare_state(state), "questions": qs}
        cached = self._cache_get(payload)
        if cached is not None:
            with self._lock:
                self.cache_hits += 1
            return Answers(
                {k: _parse_answer(v) for k, v in cached["answers"].items()},
                cached.get("usage", {}), cached, 0.0,
            )

        t0 = time.perf_counter()
        raw = self._post(payload)
        latency = time.perf_counter() - t0

        usage = raw.get("usage", {})
        with self._lock:
            self.calls += 1
            self.total_cost += float(usage.get("cost", 0.0))
            self.total_input_tokens += int(usage.get("input_tokens", 0))

        self._cache_put(payload, raw)
        return Answers({k: _parse_answer(v) for k, v in raw["answers"].items()}, usage, raw, latency)

    # ---------------- пакетная обработка ----------------

    def map(
        self,
        states: Iterable[Any],
        questions: Mapping[str, dict],
        concurrency: int | None = None,
        on_result: Callable[[int, Any, Answers], None] | None = None,
    ) -> list[Answers]:
        """Задать один набор вопросов множеству состояний, параллельно.

        Порядок результатов соответствует порядку входа. Это рабочая лошадка
        для map-reduce по данным: фильтрация логов, скрининг, разметка.
        """
        states = list(states)
        n = concurrency or self.concurrency
        results: list[Answers | None] = [None] * len(states)

        def run(i: int) -> None:
            results[i] = self.ask(states[i], questions)
            if on_result:
                on_result(i, states[i], results[i])

        with ThreadPoolExecutor(max_workers=n) as pool:
            list(pool.map(run, range(len(states))))
        return results  # type: ignore[return-value]

    def filter(
        self,
        states: Iterable[Any],
        instructions: str,
        threshold: float = 0.5,
        **kw: Any,
    ) -> list[Any]:
        """Отфильтровать коллекцию одним булевым вопросом. Дешёвый предфильтр
        перед дорогой моделью."""
        states = list(states)
        answers = self.map(states, {"keep": noul(instructions)}, **kw)
        return [s for s, a in zip(states, answers) if a.keep.p >= threshold]

    # ---------------- «умный if» ----------------

    def predicate(self, instructions: str, threshold: float = 0.5, **crit: str) -> Callable[[Any], bool]:
        """Сделать из вопроса обычную питоновскую функцию-предикат.

        >>> is_spam = jev.predicate("Это спам?")
        >>> if is_spam(message): ...
        """
        q = {"p": noul(instructions, **crit)}

        def check(state: Any) -> bool:
            return self.ask(state, q).p.p >= threshold

        check.__doc__ = f"Jev-предикат: {instructions}"
        return check

    # ---------------- транспорт ----------------

    def _post(self, payload: dict) -> dict:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        last: Exception | None = None
        for attempt in range(self.retries):
            req = urllib.request.Request(
                self.base_url,
                data=body,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "X-Title": "jev.py",
                },
            )
            try:
                with urllib.request.urlopen(req, timeout=self.timeout, context=self._ssl) as f:
                    return json.loads(f.read().decode("utf-8"))
            except urllib.error.HTTPError as e:
                text = e.read().decode("utf-8", "replace")
                # 4xx (кроме 429) — наша вина, повтор не поможет
                if e.code < 500 and e.code != 429:
                    raise JevError(e.code, text) from None
                last = JevError(e.code, text)
            except Exception as e:  # сеть, таймаут
                last = e
            if attempt < self.retries - 1:
                time.sleep(0.4 * (2 ** attempt))
        raise last  # type: ignore[misc]

    # ---------------- кэш ----------------

    def _key(self, payload: dict) -> str:
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
        return hashlib.sha256(blob).hexdigest()[:32]

    def _cache_get(self, payload: dict) -> dict | None:
        if not self.cache_dir:
            return None
        path = os.path.join(self.cache_dir, self._key(payload) + ".json")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        return None

    def _cache_put(self, payload: dict, raw: dict) -> None:
        if not self.cache_dir:
            return
        path = os.path.join(self.cache_dir, self._key(payload) + ".json")
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(raw, f, ensure_ascii=False)
        os.replace(tmp, path)

    # ---------------- прочее ----------------

    def stats(self) -> str:
        return (f"{self.calls} запросов, {self.total_input_tokens} входных токенов, "
                f"${self.total_cost:.6f}, попаданий в кэш: {self.cache_hits}")


def _prepare_state(state: Any) -> Any:
    """API принимает строку, словарь или список. Остальное — сериализуем."""
    if isinstance(state, str):
        return state
    if isinstance(state, Mapping):
        return {str(k): (v if isinstance(v, (str, int, float, bool, type(None))) else
                         json.dumps(v, ensure_ascii=False)) for k, v in state.items()}
    if isinstance(state, (list, tuple)):
        return list(state)
    return str(state)


def _ssl_context() -> ssl.SSLContext:
    """На macOS системный Python часто идёт без корневых сертификатов."""
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()
