"""
Захисні механізми ReAct-агента:
- max_steps (обмеження ітерацій циклу LLM–tools–LLM),
- timeout (загальний тайм-аут виконання через time.monotonic()),
- LoopDetector (детекція повторюваних tool calls),
- обробка rate limit (HTTP 429) для стабільної роботи з API-провайдерами.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class SafetyConfig:
    """Конфігурація лімітів безпеки агента."""

    max_steps: int = 10
    timeout_seconds: float = 120.0
    max_repeats: int = 3
    max_rate_limit_retries: int = 3


class LoopDetector:
    """Детектор зациклення агента на однакових викликах інструментів."""

    def __init__(self, max_repeats: int = 3) -> None:
        if max_repeats < 2:
            raise ValueError("max_repeats має бути >= 2.")
        self.max_repeats = max_repeats
        self.recent_calls: list[str] = []
        self.call_signatures: list[str] = []

    def _make_hash(self, tool_name: str, args: dict[str, Any]) -> str:
        """Обчислює детермінований MD5-хеш пари (tool_name, args)."""
        canonical = json.dumps(
            {"tool": tool_name, "args": args or {}},
            sort_keys=True,
            ensure_ascii=False,
            default=str,
        )
        return hashlib.md5(canonical.encode("utf-8")).hexdigest()

    def check(self, tool_name: str, args: dict[str, Any]) -> bool:
        """Фіксує виклик та повертає True, якщо виявлено зациклення."""
        call_hash = self._make_hash(tool_name, args)
        self.recent_calls.append(call_hash)
        self.call_signatures.append(f"{tool_name}({args})")

        if len(self.recent_calls) >= self.max_repeats:
            last_n = self.recent_calls[-self.max_repeats :]
            if len(set(last_n)) == 1:
                return True
        return False

    def reset(self) -> None:
        """Очищає історію викликів перед новим запуском агента."""
        self.recent_calls.clear()
        self.call_signatures.clear()


class SafetyGuard:
    """Менеджер безпеки одного запуску (run) ReAct-агента."""

    def __init__(self, config: SafetyConfig | None = None) -> None:
        self.config = config or SafetyConfig()
        self.loop_detector = LoopDetector(max_repeats=self.config.max_repeats)
        self.start_time: float = time.monotonic()

    def reset(self) -> None:
        """Скидає таймер та історію викликів перед новим запитом."""
        self.start_time = time.monotonic()
        self.loop_detector.reset()

    def elapsed_seconds(self) -> float:
        """Повертає кількість секунд, що минули з початку запуску."""
        return time.monotonic() - self.start_time

    def remaining_seconds(self) -> float:
        """Повертає залишок бюджету часу до timeout."""
        return max(0.0, self.config.timeout_seconds - self.elapsed_seconds())

    def is_max_steps_reached(self, current_step: int) -> bool:
        """Перевіряє, чи досягнуто ліміт кроків."""
        return current_step >= self.config.max_steps

    def is_timed_out(self) -> bool:
        """Перевіряє, чи вичерпано загальний тайм-аут виконання."""
        return self.elapsed_seconds() >= self.config.timeout_seconds

    def check_tool_calls_for_loop(self, tool_calls: list[dict[str, Any]]) -> str | None:
        """Перевіряє список tool_calls на повторюваність.

        Returns:
            Опис зацикленого виклику або None, якщо зациклення немає.
        """
        for call in tool_calls:
            name = str(call.get("name", "unknown"))
            args = call.get("args", {}) or {}
            if self.loop_detector.check(name, args):
                return f"{name}({json.dumps(args, ensure_ascii=False, sort_keys=True)})"
        return None

    @staticmethod
    def _extract_retry_delay(error_text: str, default_delay: float = 20.0) -> float:
        """Витягує рекомендовану затримку з повідомлення Gemini 429."""
        match = re.search(r"(?:retry in|retryDelay['\"]?:\s*['\"]?)(\d+(?:\.\d+)?)s", error_text, re.IGNORECASE)
        if match:
            return float(match.group(1)) + 2.0
        return default_delay

    def invoke_with_rate_limit_retry(self, fn: Callable[[], Any]) -> Any:
        """Виконує виклик LLM із повторною спробою при HTTP 429 у межах timeout."""
        for attempt in range(self.config.max_rate_limit_retries + 1):
            if self.is_timed_out():
                raise TimeoutError(
                    f"Перевищено загальний тайм-аут виконання ({self.config.timeout_seconds:.1f} с)."
                )
            try:
                return fn()
            except Exception as exc:
                err_str = str(exc)
                is_rate_limit = any(code in err_str for code in ("429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE"))
                if not is_rate_limit or attempt >= self.config.max_rate_limit_retries:
                    raise

                delay = self._extract_retry_delay(err_str, default_delay=15.0 * (attempt + 1))
                remaining = self.remaining_seconds()
                if remaining <= 2.0 or delay >= remaining:
                    raise TimeoutError(
                        f"Недостатньо часу ({remaining:.1f} с) для очікування rate limit 429 (потрібно {delay:.1f} с)."
                    ) from exc

                sleep_time = delay
                print(
                    f"  [safety/rate-limit] Отримано 429 від API. "
                    f"Пауза {sleep_time:.1f}с перед спробою {attempt + 2}/{self.config.max_rate_limit_retries + 1}...",
                    flush=True,
                )
                time.sleep(sleep_time)


# ── Автономні unit-тести захисних механізмів ──────────────────────────────────
if __name__ == "__main__":
    print("=== Unit-тест LoopDetector ===")
    detector = LoopDetector(max_repeats=3)
    assert detector.check("estimate_gpu_vram", {"model_name": "yolo11m", "batch_size": 16}) is False
    assert detector.check("estimate_gpu_vram", {"batch_size": 16, "model_name": "yolo11m"}) is False
    assert detector.check("estimate_gpu_vram", {"model_name": "yolo11m", "batch_size": 16}) is True
    print("[OK] LoopDetector виявив 3 однакові виклики поспіль (незалежно від порядку ключів).")

    detector.reset()
    assert detector.check("estimate_gpu_vram", {"batch_size": 16}) is False
    assert detector.check("estimate_gpu_vram", {"batch_size": 8}) is False
    assert detector.check("estimate_gpu_vram", {"batch_size": 4}) is False
    print("[OK] LoopDetector НЕ спрацьовує, коли аргументи змінюються.")

    print("\n=== Unit-тест SafetyGuard (max_steps та timeout) ===")
    guard = SafetyGuard(SafetyConfig(max_steps=3, timeout_seconds=0.2, max_repeats=3))
    assert guard.is_max_steps_reached(2) is False
    assert guard.is_max_steps_reached(3) is True
    print("[OK] max_steps спрацьовує на кроці 3.")

    assert guard.is_timed_out() is False
    time.sleep(0.25)
    assert guard.is_timed_out() is True
    print("[OK] timeout через time.monotonic() спрацьовує коректно.")