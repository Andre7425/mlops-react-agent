"""
Запуск 5 тест-кейсів різної складності для MLOps / Computer Vision ReAct-агента.
Зберігає результати у test_results.json та повну траєкторію у trajectory.json.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage

from agent import (
    AgentResponse,
    build_agent_graph,
    create_llm,
    extract_all_tool_calls,
    extract_text,
)
from logger import TrajectoryLogger
from safety import SafetyConfig

# Пауза між тестами (сек), щоб на Gemini Free Tier (5 RPM) ліміт не спотворював latency тестів
INTER_TEST_PAUSE_SEC = float(os.getenv("INTER_TEST_PAUSE_SEC", "18"))


TEST_CASES: list[dict[str, Any]] = [
    {
        "id": "TC-001",
        "complexity": "simple",
        "query": (
            "Перевір, чи поміститься навчання yolo11n у fp16 з batch_size=16 "
            "та img_size=640 на відеокарті з 6 GB VRAM."
        ),
        "expected": (
            "Виклик estimate_gpu_vram(yolo11n, train, fp16, batch_size=16, img_size=640, gpu_vram_gb=6.0); "
            "підтвердження статусу SAFE і точна оцінка VRAM (~1.7 ГБ)."
        ),
        "expected_tools": ["estimate_gpu_vram"],
    },
    {
        "id": "TC-002",
        "complexity": "medium",
        "query": (
            "Порівняй споживання VRAM для навчання yolo11m (batch_size=8, img_size=640, GPU 6 GB) "
            "у точності fp32 проти fp16 та скажи, який варіант безпечний."
        ),
        "expected": (
            "Два виклики estimate_gpu_vram (для fp32 і fp16); порівняння точних ГБ VRAM "
            "та висновок, що fp16 безпечний, а fp32 має високий ризик OOM."
        ),
        "expected_tools": ["estimate_gpu_vram"],
    },
    {
        "id": "TC-003",
        "complexity": "medium",
        "query": (
            "Хочу тренувати yolo11l у 1024x1024 на RTX 2060 (6 GB VRAM) з ефективним батчем 16. "
            "Спочатку оціни VRAM для batch_size=8 у fp16, а потім підбери безпечну конфігурацію."
        ),
        "expected": (
            "Виклик estimate_gpu_vram (OOM_EXCEEDED > 6 GB) + recommend_training_config "
            "(micro_batch_size=4, gradient_accumulation_steps=4, fp16, ~4.26 GB)."
        ),
        "expected_tools": ["estimate_gpu_vram", "recommend_training_config"],
    },
    {
        "id": "TC-004",
        "complexity": "complex",
        "query": (
            "Проведи строгий QA-аудит партії DRONE-07: 4000 зображень, перевірено 450, "
            "mean_iou=0.72, honeypot_accuracy=0.84, missing_label_rate=0.06. "
            "Якщо партію відхилять, обов'язково розрахуй через інструмент бюджет повної перерозмітки "
            "для bbox (середньо 8 об'єктів/кадр, ставка $11/год, 15% QA) у двох варіантах: "
            "без pre-labeling та з pre-labeling."
        ),
        "expected": (
            "audit_annotation_batch -> REJECTED (4000 кадрів на rework); "
            "два виклики estimate_labeling_budget (~$758.97 без pre-labeling vs ~$326.30 з pre-labeling)."
        ),
        "expected_tools": ["audit_annotation_batch", "estimate_labeling_budget"],
    },
    {
        "id": "TC-005",
        "complexity": "complex",
        "query": (
            "Планую навчати efficientnet_b4 (img_size=640) на GPU з 8 GB VRAM. "
            "Перевір, чи можна запустити навчання в int4 з batch_size=16; якщо ні — підбери "
            "безпечну конфігурацію для ефективного batch_size=32 (без bf16) і додатково порахуй "
            "бюджет розмітки датасету на 3000 зображень типу classification (ставка $9/год, 10% QA)."
        ),
        "expected": (
            "Коректна обробка заборони int4 у train (через валідацію або правило), "
            "підбір конфігу через recommend_training_config + розрахунок бюджету через estimate_labeling_budget."
        ),
        "expected_tools": ["recommend_training_config", "estimate_labeling_budget"],
    },
]


def evaluate_status(
    stop_reason: str | None,
    tool_calls: list[str],
    expected_tools: list[str],
    error: str | None,
) -> str:
    """Класифікує результат тест-кейсу: success / partial / error."""
    if error:
        return "error"
    if stop_reason in ("max_steps", "timeout", "loop_detected"):
        return "partial"
    missing = [t for t in expected_tools if t not in tool_calls]
    if missing:
        return "partial"
    return "success"


def run_test_suite() -> None:
    """Проганяє всі тест-кейси та зберігає test_results.json і trajectory.json."""
    llm = create_llm()
    safety_cfg = SafetyConfig(max_steps=10, timeout_seconds=120.0, max_repeats=3)

    results: list[dict[str, Any]] = []
    combined_trajectory: list[dict[str, Any]] = []
    suite_start = time.monotonic()

    for idx, tc in enumerate(TEST_CASES):
        test_id = tc["id"]
        print(f"\n{'=' * 75}")
        print(f"[{idx + 1}/{len(TEST_CASES)}] {test_id} ({tc['complexity']}): {tc['query'][:75]}...")
        print(f"{'=' * 75}", flush=True)

        traj_logger = TrajectoryLogger(max_field_length=500)
        traj_logger.reset(metadata={"test_id": test_id, "complexity": tc["complexity"]})

        app, _, _ = build_agent_graph(
            llm=llm,
            safety_config=safety_cfg,
            trajectory_logger=traj_logger,
            verbose=True,
        )

        t0 = time.monotonic()
        error_text: str | None = None
        stop_reason: str | None = None
        steps = 0
        tool_calls: list[str] = []
        actual_text = ""
        confidence: float | None = None
        sources: list[str] = []

        try:
            state = app.invoke(
                {
                    "messages": [HumanMessage(content=tc["query"])],
                    "step_count": 0,
                    "stop_reason": None,
                    "structured_response": None,
                }
            )
            steps = int(state.get("step_count", 0))
            stop_reason = state.get("stop_reason")
            tool_calls = extract_all_tool_calls(state.get("messages", []))

            structured: AgentResponse | None = state.get("structured_response")
            if structured is not None:
                actual_text = structured.answer
                confidence = structured.confidence
                sources = structured.sources
            else:
                actual_text = extract_text(state["messages"][-1].content)
        except Exception as exc:
            error_text = f"{type(exc).__name__}: {exc}"
            actual_text = f"Error: {error_text}"
            steps = -1

        elapsed_ms = int((time.monotonic() - t0) * 1000)
        status = evaluate_status(stop_reason, tool_calls, tc["expected_tools"], error_text)

        # Додаємо кроки цього тесту в спільну траєкторію з тегом test_id
        for step_entry in traj_logger.steps:
            combined_trajectory.append({"test_id": test_id, **step_entry})

        results.append(
            {
                "test_id": test_id,
                "complexity": tc["complexity"],
                "query": tc["query"],
                "expected": tc["expected"],
                "expected_tools": tc["expected_tools"],
                "actual": actual_text,
                "confidence": confidence,
                "sources": sources,
                "status": status,
                "steps": steps,
                "tool_calls": tool_calls,
                "elapsed_ms": elapsed_ms,
                "stop_reason": stop_reason,
                "hit_max_steps": stop_reason == "max_steps",
                "hit_timeout": stop_reason == "timeout",
                "hit_loop_detector": stop_reason == "loop_detected",
            }
        )

        print(
            f"-> {test_id}: status={status} | steps={steps} | "
            f"tools={tool_calls} | time={elapsed_ms} мс | stop_reason={stop_reason}",
            flush=True,
        )

        # Пауза перед наступним тестом, щоб не вичерпати Free Tier RPM
        if idx < len(TEST_CASES) - 1 and INTER_TEST_PAUSE_SEC > 0:
            print(f"   Пауза {INTER_TEST_PAUSE_SEC:.0f}с для відновлення квоти Gemini API...", flush=True)
            time.sleep(INTER_TEST_PAUSE_SEC)

    # Збереження test_results.json
    results_path = Path("test_results.json")
    with results_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # Збереження об'єднаної повної траєкторії у trajectory.json
    trajectory_path = Path("trajectory.json")
    trajectory_payload = {
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "model": os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        "safety_config": {
            "max_steps": safety_cfg.max_steps,
            "timeout_seconds": safety_cfg.timeout_seconds,
            "max_repeats": safety_cfg.max_repeats,
        },
        "total_tests": len(results),
        "total_steps": len(combined_trajectory),
        "total_time_ms": int((time.monotonic() - suite_start) * 1000),
        "trajectory": combined_trajectory,
    }
    with trajectory_path.open("w", encoding="utf-8") as f:
        json.dump(trajectory_payload, f, ensure_ascii=False, indent=2)

    # Підсумкова таблиця в консолі
    print(f"\n{'=' * 75}")
    print(f"Збережено: {results_path} ({len(results)} тестів) та {trajectory_path} ({len(combined_trajectory)} кроків)")
    print(f"{'=' * 75}")
    for r in results:
        print(
            f"  {r['test_id']} [{r['complexity']:7s}] | status={r['status']:7s} | "
            f"steps={r['steps']} | tools={r['tool_calls']} | {r['elapsed_ms']} мс"
        )


if __name__ == "__main__":
    run_test_suite()