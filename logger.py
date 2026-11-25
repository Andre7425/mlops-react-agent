"""
Модуль логування повної траєкторії виконання ReAct-агента у JSON.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class TrajectoryLogger:
    """Логування кожного кроку виконання графа LangGraph у JSON-файл."""

    def __init__(self, max_field_length: int = 500) -> None:
        self.max_field_length = max_field_length
        self.steps: list[dict[str, Any]] = []
        self.start_time: float = time.monotonic()
        self.run_metadata: dict[str, Any] = {}

    def reset(self, metadata: dict[str, Any] | None = None) -> None:
        """Скидає лог перед новим запуском агента."""
        self.steps = []
        self.start_time = time.monotonic()
        self.run_metadata = metadata or {}

    def _truncate(self, value: Any) -> str:
        """Перетворює значення на рядок та обрізає до max_field_length символів."""
        text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
        if len(text) > self.max_field_length:
            return text[: self.max_field_length] + "..."
        return text

    def log_step(
        self,
        step_number: int,
        node_name: str,
        input_data: Any,
        output_data: Any,
        tool_calls: list[Any] | None = None,
        duration_ms: int = 0,
    ) -> None:
        """Записує один крок траєкторії."""
        self.steps.append(
            {
                "step_number": step_number,
                "node_name": node_name,
                "input": self._truncate(input_data),
                "output": self._truncate(output_data),
                "tool_calls": tool_calls or [],
                "timestamp": datetime.now(tz=timezone.utc).isoformat(),
                "duration_ms": int(duration_ms),
                "elapsed_ms": int((time.monotonic() - self.start_time) * 1000),
            }
        )

    def to_dict(self) -> dict[str, Any]:
        """Повертає повну структуру траєкторії як словник."""
        return {
            "metadata": self.run_metadata,
            "total_steps": len(self.steps),
            "total_time_ms": int((time.monotonic() - self.start_time) * 1000),
            "trajectory": self.steps,
        }

    def save(self, filepath: str | Path = "trajectory.json") -> Path:
        """Зберігає траєкторію у JSON-файл з підтримкою кирилиці."""
        path = Path(filepath)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        print(f"Траєкторію збережено: {path} ({len(self.steps)} кроків)", flush=True)
        return path