"""
Доменні інструменти (tools) для MLOps / Computer Vision ReAct-агента.
Реалізують сувору валідацію вхідних параметрів через Pydantic v2
та повертають структуровані JSON-відповіді для надійного парсингу LLM.
"""

from __future__ import annotations

import json
import math
import re
from typing import Literal

from langchain_core.tools import tool
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# ── Довідник архітектур Computer Vision ───────────────────────────────────────
# params_m: кількість параметрів у мільйонах
# act_mb_per_img_640_fp32: орієнтовний обсяг активацій (МБ) на 1 зображення 640x640 у FP32 при навчанні
CV_MODEL_SPECS: dict[str, dict[str, float | str]] = {
    "yolo11n": {"params_m": 2.6, "act_mb_per_img_640_fp32": 115.0, "task": "detection"},
    "yolo11s": {"params_m": 9.4, "act_mb_per_img_640_fp32": 240.0, "task": "detection"},
    "yolo11m": {"params_m": 20.1, "act_mb_per_img_640_fp32": 430.0, "task": "detection"},
    "yolo11l": {"params_m": 25.3, "act_mb_per_img_640_fp32": 580.0, "task": "detection"},
    "yolov8m": {"params_m": 25.9, "act_mb_per_img_640_fp32": 460.0, "task": "detection"},
    "rtdetr_l": {"params_m": 32.0, "act_mb_per_img_640_fp32": 650.0, "task": "detection"},
    "efficientnet_b0": {"params_m": 5.3, "act_mb_per_img_640_fp32": 180.0, "task": "classification"},
    "efficientnet_b4": {"params_m": 19.3, "act_mb_per_img_640_fp32": 490.0, "task": "classification"},
    "resnet50": {"params_m": 25.6, "act_mb_per_img_640_fp32": 360.0, "task": "classification"},
    "sam_vit_b": {"params_m": 91.0, "act_mb_per_img_640_fp32": 820.0, "task": "segmentation"},
    "sam_vit_h": {"params_m": 636.0, "act_mb_per_img_640_fp32": 2100.0, "task": "segmentation"},
}

BYTES_PER_PARAM: dict[str, float] = {
    "fp32": 4.0,
    "fp16": 2.0,
    "bf16": 2.0,
    "int4": 0.5,
}


def _compute_vram_breakdown(
    model_name: str,
    mode: Literal["train", "inference"],
    precision: Literal["fp32", "fp16", "bf16", "int4"],
    batch_size: int,
    img_size: int,
    gpu_vram_gb: float,
) -> dict:
    """Внутрішня детермінована функція розрахунку VRAM (спільна для двох tools)."""
    spec = CV_MODEL_SPECS[model_name]
    params_m = float(spec["params_m"])
    base_act_mb = float(spec["act_mb_per_img_640_fp32"])

    bytes_pp = BYTES_PER_PARAM[precision]
    # Ваги моделі в ГБ
    weights_gb = (params_m * 1e6 * bytes_pp) / (1024**3)

    # Масштабування активацій квадратично до роздільної здатності зображення
    resolution_scale = (img_size / 640.0) ** 2
    precision_act_scale = 0.55 if precision in ("fp16", "bf16") else (0.35 if precision == "int4" else 1.0)

    if mode == "train":
        # AdamW зберігає FP32 master weights + 2 моменти + градієнти (~16 байт/параметр для AMP або 12 для FP32)
        opt_and_grad_bytes = 16.0 if precision in ("fp16", "bf16") else 12.0
        optimizer_and_grad_gb = (params_m * 1e6 * opt_and_grad_bytes) / (1024**3)
        activations_gb = (base_act_mb * batch_size * resolution_scale * precision_act_scale) / 1024.0
        cuda_overhead_gb = 0.65
    else:
        optimizer_and_grad_gb = 0.0
        # При інференсі проміжні графи активацій не зберігаються для backprop (~22% від train)
        activations_gb = (base_act_mb * 0.22 * batch_size * resolution_scale * precision_act_scale) / 1024.0
        cuda_overhead_gb = 0.45

    total_vram_gb = weights_gb + optimizer_and_grad_gb + activations_gb + cuda_overhead_gb
    utilization_pct = (total_vram_gb / gpu_vram_gb) * 100.0
    fits = total_vram_gb <= gpu_vram_gb

    if utilization_pct <= 80.0:
        status = "SAFE"
    elif utilization_pct <= 95.0:
        status = "TIGHT_HEADROOM"
    elif utilization_pct <= 100.0:
        status = "HIGH_OOM_RISK"
    else:
        status = "OOM_EXCEEDED"

    return {
        "model_name": model_name,
        "mode": mode,
        "precision": precision,
        "batch_size": batch_size,
        "img_size": img_size,
        "gpu_vram_gb": round(gpu_vram_gb, 2),
        "breakdown_gb": {
            "weights_gb": round(weights_gb, 3),
            "optimizer_and_grad_gb": round(optimizer_and_grad_gb, 3),
            "activations_gb": round(activations_gb, 3),
            "cuda_overhead_gb": round(cuda_overhead_gb, 3),
        },
        "total_vram_gb": round(total_vram_gb, 2),
        "free_vram_gb": round(gpu_vram_gb - total_vram_gb, 2),
        "utilization_pct": round(utilization_pct, 1),
        "fits_in_gpu": fits,
        "status": status,
    }


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 1: Оцінка споживання GPU VRAM
# ══════════════════════════════════════════════════════════════════════════════

class EstimateVramInput(BaseModel):
    """Схема вхідних параметрів для оцінки споживання відеопам'яті GPU."""

    model_config = ConfigDict(str_strip_whitespace=True)

    model_name: str = Field(
        description=(
            "Назва CV-моделі. Доступні значення: "
            + ", ".join(sorted(CV_MODEL_SPECS.keys()))
        )
    )
    mode: Literal["train", "inference"] = Field(
        default="train",
        description="Режим запуску: 'train' для навчання або 'inference' для інференсу.",
    )
    precision: Literal["fp32", "fp16", "bf16", "int4"] = Field(
        default="fp16",
        description="Точність обчислень: fp32, fp16, bf16 або int4 (int4 лише для inference).",
    )
    batch_size: int = Field(
        default=16,
        description="Розмір батчу (ціле число від 1 до 512).",
    )
    img_size: int = Field(
        default=640,
        description="Роздільна здатність квадратного входу в пікселях (від 128 до 2048, кратна 32).",
    )
    gpu_vram_gb: float = Field(
        default=6.0,
        description="Доступний обсяг відеопам'яті GPU в ГБ (від 2.0 до 192.0).",
    )

    @field_validator("model_name")
    @classmethod
    def validate_model_name(cls, v: str) -> str:
        normalized = v.lower().replace("-", "_").replace(" ", "_")
        if normalized not in CV_MODEL_SPECS:
            allowed = ", ".join(sorted(CV_MODEL_SPECS.keys()))
            raise ValueError(
                f"Невідома модель '{v}'. Оберіть одну з підтримуваних: {allowed}."
            )
        return normalized

    @field_validator("batch_size")
    @classmethod
    def validate_batch_size(cls, v: int) -> int:
        if not 1 <= v <= 512:
            raise ValueError("batch_size має бути в діапазоні від 1 до 512.")
        return v

    @field_validator("img_size")
    @classmethod
    def validate_img_size(cls, v: int) -> int:
        if not 128 <= v <= 2048:
            raise ValueError("img_size має бути в діапазоні від 128 до 2048 пікселів.")
        if v % 32 != 0:
            raise ValueError(
                f"img_size ({v}) має бути кратним 32 (наприклад, 320, 640, 1024, 1280)."
            )
        return v

    @field_validator("gpu_vram_gb")
    @classmethod
    def validate_gpu_vram(cls, v: float) -> float:
        if not 2.0 <= v <= 192.0:
            raise ValueError("gpu_vram_gb має бути в межах від 2.0 до 192.0 ГБ.")
        return v

    @model_validator(mode="after")
    def validate_precision_for_mode(self) -> "EstimateVramInput":
        if self.mode == "train" and self.precision == "int4":
            raise ValueError(
                "Точність 'int4' не підтримується для повноцінного навчання CV-моделей. "
                "Використайте 'fp16', 'bf16' або 'fp32'."
            )
        return self


@tool(args_schema=EstimateVramInput)
def estimate_gpu_vram(
    model_name: str,
    mode: Literal["train", "inference"] = "train",
    precision: Literal["fp32", "fp16", "bf16", "int4"] = "fp16",
    batch_size: int = 16,
    img_size: int = 640,
    gpu_vram_gb: float = 6.0,
) -> str:
    """Розрахувати оцінку споживання GPU VRAM для навчання або інференсу CV-моделі.

    Використовуйте цей інструмент, коли потрібно перевірити, чи поміститься модель
    (YOLO, ResNet, EfficientNet, RT-DETR, SAM) у відеопам'ять конкретної GPU
    при заданих batch_size, img_size та precision.

    Returns:
        JSON-рядок із детальним розкладом VRAM, відсотком завантаження та статусом ризику OOM.
    """
    result = _compute_vram_breakdown(
        model_name=model_name,
        mode=mode,
        precision=precision,
        batch_size=batch_size,
        img_size=img_size,
        gpu_vram_gb=gpu_vram_gb,
    )
    return json.dumps(result, ensure_ascii=False)


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 2: Підбір безпечної конфігурації навчання під ліміт GPU
# ══════════════════════════════════════════════════════════════════════════════

class RecommendConfigInput(BaseModel):
    """Схема параметрів для підбору безпечної конфігурації навчання."""

    model_config = ConfigDict(str_strip_whitespace=True)

    model_name: str = Field(
        description="Назва CV-моделі зі списку підтримуваних (наприклад, 'yolo11m', 'yolo11l')."
    )
    gpu_vram_gb: float = Field(
        description="Обсяг відеопам'яті цільової GPU в ГБ (наприклад, 6.0, 12.0, 24.0)."
    )
    img_size: int = Field(
        default=640,
        description="Бажана роздільна здатність навчання (128–2048, кратна 32).",
    )
    target_effective_batch_size: int = Field(
        default=16,
        description="Цільовий ефективний batch_size (від 1 до 256), який потрібно досягти через gradient accumulation.",
    )
    prefer_bf16: bool = Field(
        default=False,
        description="True для GPU архітектури Ampere+ (RTX 30xx/40xx, A100/L4), False для Turing (RTX 2060/T4).",
    )

    @field_validator("model_name")
    @classmethod
    def validate_model_name(cls, v: str) -> str:
        normalized = v.lower().replace("-", "_").replace(" ", "_")
        if normalized not in CV_MODEL_SPECS:
            allowed = ", ".join(sorted(CV_MODEL_SPECS.keys()))
            raise ValueError(f"Невідома модель '{v}'. Доступні: {allowed}.")
        return normalized

    @field_validator("gpu_vram_gb")
    @classmethod
    def validate_vram(cls, v: float) -> float:
        if not 2.0 <= v <= 192.0:
            raise ValueError("gpu_vram_gb має бути в межах від 2.0 до 192.0 ГБ.")
        return v

    @field_validator("img_size")
    @classmethod
    def validate_img_size(cls, v: int) -> int:
        if not 128 <= v <= 2048 or v % 32 != 0:
            raise ValueError("img_size має бути в діапазоні 128–2048 та кратним 32.")
        return v

    @field_validator("target_effective_batch_size")
    @classmethod
    def validate_eff_batch(cls, v: int) -> int:
        if not 1 <= v <= 256:
            raise ValueError("target_effective_batch_size має бути від 1 до 256.")
        return v


@tool(args_schema=RecommendConfigInput)
def recommend_training_config(
    model_name: str,
    gpu_vram_gb: float,
    img_size: int = 640,
    target_effective_batch_size: int = 16,
    prefer_bf16: bool = False,
) -> str:
    """Підібрати безпечну конфігурацію навчання CV-моделі під обмеження GPU VRAM.

    Використовуйте цей інструмент, коли потрібно знайти оптимальний micro_batch_size,
    gradient_accumulation_steps та precision, щоб навчання гарантовано поміщалося
    у <=85% відеопам'яті без OOM.

    Returns:
        JSON-рядок з рекомендованими параметрами навчання та прогнозованим VRAM.
    """
    precision: Literal["bf16", "fp16"] = "bf16" if prefer_bf16 else "fp16"
    safe_limit_gb = gpu_vram_gb * 0.85

    # Кандидати micro_batch_size (ступені двійки + точне значення target)
    candidates = sorted(
        {b for b in (1, 2, 4, 8, 16, 32, 64, 128, target_effective_batch_size) if b <= target_effective_batch_size},
        reverse=True,
    )

    chosen_batch = None
    chosen_vram = None
    for b in candidates:
        est = _compute_vram_breakdown(
            model_name=model_name,
            mode="train",
            precision=precision,
            batch_size=b,
            img_size=img_size,
            gpu_vram_gb=gpu_vram_gb,
        )
        if est["total_vram_gb"] <= safe_limit_gb:
            chosen_batch = b
            chosen_vram = est
            break

    if chosen_batch is None or chosen_vram is None:
        # Навіть batch=1 перевищує безпечний поріг -> пропонуємо зменшити img_size або модель
        fallback = _compute_vram_breakdown(
            model_name=model_name,
            mode="train",
            precision=precision,
            batch_size=1,
            img_size=img_size,
            gpu_vram_gb=gpu_vram_gb,
        )
        reduced_img = max(320, (img_size // 2) - ((img_size // 2) % 32))
        payload = {
            "feasible_at_requested_resolution": False,
            "model_name": model_name,
            "gpu_vram_gb": gpu_vram_gb,
            "safe_vram_limit_gb": round(safe_limit_gb, 2),
            "min_batch1_vram_gb": fallback["total_vram_gb"],
            "recommendations": [
                f"Зменшити img_size з {img_size} до {reduced_img} пікселів.",
                "Перейти на легшу архітектуру (наприклад, yolo11n або yolo11s).",
                "Увімкнути gradient checkpointing або використати хмарну GPU (>=16 GB VRAM).",
            ],
        }
        return json.dumps(payload, ensure_ascii=False)

    grad_accum = math.ceil(target_effective_batch_size / chosen_batch)
    actual_effective_batch = chosen_batch * grad_accum

    payload = {
        "feasible_at_requested_resolution": True,
        "model_name": model_name,
        "gpu_vram_gb": gpu_vram_gb,
        "safe_vram_limit_gb": round(safe_limit_gb, 2),
        "recommended_config": {
            "precision": precision,
            "img_size": img_size,
            "micro_batch_size": chosen_batch,
            "gradient_accumulation_steps": grad_accum,
            "actual_effective_batch_size": actual_effective_batch,
            "num_workers": 4 if gpu_vram_gb <= 8.0 else 8,
            "pin_memory": True,
        },
        "expected_vram_gb": chosen_vram["total_vram_gb"],
        "expected_utilization_pct": chosen_vram["utilization_pct"],
        "notes": (
            "Збережено >=15% запасу VRAM для стабільності та уникнення OOM-спайків під час валідації."
        ),
    }
    return json.dumps(payload, ensure_ascii=False)


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 3: QA-аудит партії розмітки датасету (DataOps)
# ══════════════════════════════════════════════════════════════════════════════

class AuditAnnotationBatchInput(BaseModel):
    """Схема вхідних даних для QA-аудиту партії розмічених зображень."""

    model_config = ConfigDict(str_strip_whitespace=True)

    batch_id: str = Field(
        description="Ідентифікатор партії розмітки (наприклад, 'BATCH-042')."
    )
    total_images: int = Field(
        description="Загальна кількість зображень у партії (> 0)."
    )
    reviewed_images: int = Field(
        description="Кількість зображень, перевірених під час QA-аудиту (> 0 і <= total_images)."
    )
    mean_iou: float = Field(
        description="Середній IoU між розміткою анотатора та еталоном QA (від 0.0 до 1.0)."
    )
    honeypot_accuracy: float = Field(
        description="Точність на контрольних (honeypot) зображеннях (від 0.0 до 1.0)."
    )
    missing_label_rate: float = Field(
        description="Частка пропущених об'єктів (False Negatives) у партії (від 0.0 до 1.0)."
    )
    strict_mode: bool = Field(
        default=False,
        description="True для критичних CV-пайплайнів (підвищені пороги IoU >= 0.85, honeypot >= 0.92).",
    )

    @field_validator("batch_id")
    @classmethod
    def validate_batch_id(cls, v: str) -> str:
        if len(v) < 3 or not re.match(r"^[A-Za-z0-9_-]+$", v):
            raise ValueError(
                "batch_id має містити мінімум 3 символи (латиниця, цифри, '-' або '_')."
            )
        return v.upper()

    @field_validator("total_images", "reviewed_images")
    @classmethod
    def validate_positive_counts(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("Кількість зображень має бути більшою за 0.")
        return v

    @field_validator("mean_iou", "honeypot_accuracy", "missing_label_rate")
    @classmethod
    def validate_unit_interval(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("Метрики mean_iou, honeypot_accuracy та missing_label_rate мають бути в [0.0, 1.0].")
        return v

    @model_validator(mode="after")
    def validate_review_subset(self) -> "AuditAnnotationBatchInput":
        if self.reviewed_images > self.total_images:
            raise ValueError(
                f"reviewed_images ({self.reviewed_images}) не може перевищувати total_images ({self.total_images})."
            )
        return self


@tool(args_schema=AuditAnnotationBatchInput)
def audit_annotation_batch(
    batch_id: str,
    total_images: int,
    reviewed_images: int,
    mean_iou: float,
    honeypot_accuracy: float,
    missing_label_rate: float,
    strict_mode: bool = False,
) -> str:
    """Провести автоматизований QA-аудит партії розмітки CV-датасету.

    Використовуйте цей інструмент для оцінки якості розміченої партії за метриками
    IoU, honeypot-тестами та часткою пропущених об'єктів, а також для отримання
    рішення щодо прийняття або відправки на перерозмітку (rework).

    Returns:
        JSON-рядок із вердиктом QA, переліком порушених порогів та обсягом перерозмітки.
    """
    # SLA-пороги якості
    thresholds = (
        {"min_iou": 0.85, "min_honeypot": 0.92, "max_missing": 0.02, "min_sample_pct": 10.0}
        if strict_mode
        else {"min_iou": 0.75, "min_honeypot": 0.85, "max_missing": 0.05, "min_sample_pct": 5.0}
    )

    sample_coverage_pct = (reviewed_images / total_images) * 100.0
    failed_gates: list[str] = []

    if sample_coverage_pct < thresholds["min_sample_pct"]:
        failed_gates.append(
            f"Недостатня вибірка QA: {sample_coverage_pct:.1f}% < {thresholds['min_sample_pct']}%"
        )
    if mean_iou < thresholds["min_iou"]:
        failed_gates.append(f"Низький mean_iou: {mean_iou:.2f} < {thresholds['min_iou']}")
    if honeypot_accuracy < thresholds["min_honeypot"]:
        failed_gates.append(
            f"Провалено honeypot-тести: {honeypot_accuracy:.2f} < {thresholds['min_honeypot']}"
        )
    if missing_label_rate > thresholds["max_missing"]:
        failed_gates.append(
            f"Зависока частка пропусків: {missing_label_rate:.2%} > {thresholds['max_missing']:.2%}"
        )

    # Оцінка вердикту та частки зображень на перерозмітку
    if not failed_gates:
        verdict = "APPROVED"
        rework_fraction = 0.0
    elif len(failed_gates) == 1 and honeypot_accuracy >= thresholds["min_honeypot"]:
        verdict = "CONDITIONAL_REWORK"
        rework_fraction = min(0.35, max(0.15, missing_label_rate * 2.5))
    else:
        verdict = "REJECTED"
        rework_fraction = 1.0

    estimated_rework_images = int(math.ceil(total_images * rework_fraction))

    payload = {
        "batch_id": batch_id,
        "mode": "strict" if strict_mode else "standard",
        "verdict": verdict,
        "sample_coverage_pct": round(sample_coverage_pct, 2),
        "thresholds": thresholds,
        "failed_gates": failed_gates,
        "estimated_rework_images": estimated_rework_images,
        "rework_pct": round(rework_fraction * 100.0, 1),
    }
    return json.dumps(payload, ensure_ascii=False)


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 4: Оцінка бюджету та часу на розмітку / перерозмітку датасету
# ══════════════════════════════════════════════════════════════════════════════

class EstimateLabelingBudgetInput(BaseModel):
    """Схема параметрів для розрахунку вартості та часу розмітки датасету."""

    model_config = ConfigDict(str_strip_whitespace=True)

    num_images: int = Field(
        description="Кількість зображень для розмітки або перерозмітки (від 1 до 1_000_000)."
    )
    avg_objects_per_image: float = Field(
        default=5.0,
        description="Середня кількість об'єктів на одному зображенні (від 0.1 до 200.0).",
    )
    annotation_type: Literal["classification", "bbox", "polygon", "keypoints"] = Field(
        default="bbox",
        description="Тип розмітки: 'classification', 'bbox', 'polygon' або 'keypoints'.",
    )
    prelabeling_assist: bool = Field(
        default=False,
        description="True, якщо використовується автоматичний pre-labeling (YOLO/SAM), що скорочує час на ~60%.",
    )
    qa_review_pct: float = Field(
        default=15.0,
        description="Відсоток зображень, що проходять QA-перевірку (від 0.0 до 100.0).",
    )
    hourly_rate_usd: float = Field(
        default=10.0,
        description="Погодинна ставка анотатора/QA в USD (від 1.0 до 500.0).",
    )

    @field_validator("num_images")
    @classmethod
    def validate_num_images(cls, v: int) -> int:
        if not 1 <= v <= 1_000_000:
            raise ValueError("num_images має бути в діапазоні від 1 до 1_000_000.")
        return v

    @field_validator("avg_objects_per_image")
    @classmethod
    def validate_avg_objects(cls, v: float) -> float:
        if not 0.1 <= v <= 200.0:
            raise ValueError("avg_objects_per_image має бути в межах від 0.1 до 200.0.")
        return v

    @field_validator("qa_review_pct")
    @classmethod
    def validate_qa_pct(cls, v: float) -> float:
        if not 0.0 <= v <= 100.0:
            raise ValueError("qa_review_pct має бути в межах від 0.0 до 100.0.")
        return v

    @field_validator("hourly_rate_usd")
    @classmethod
    def validate_hourly_rate(cls, v: float) -> float:
        if not 1.0 <= v <= 500.0:
            raise ValueError("hourly_rate_usd має бути в межах від 1.0 до 500.0 USD.")
        return v


@tool(args_schema=EstimateLabelingBudgetInput)
def estimate_labeling_budget(
    num_images: int,
    avg_objects_per_image: float = 5.0,
    annotation_type: Literal["classification", "bbox", "polygon", "keypoints"] = "bbox",
    prelabeling_assist: bool = False,
    qa_review_pct: float = 15.0,
    hourly_rate_usd: float = 10.0,
) -> str:
    """Розрахувати орієнтовний час та бюджет на розмітку або перерозмітку CV-датасету.

    Використовуйте цей інструмент для планування бюджету DataOps, оцінки вартості
    перерозмітки відхиленої партії або порівняння ручної розмітки з pre-labeling.

    Returns:
        JSON-рядок із годинами роботи, загальним бюджетом у USD, ціною за кадр та людино-днями.
    """
    # Базові нормативи часу: (секунд на відкриття кадру, секунд на 1 об'єкт)
    timing_sec: dict[str, tuple[float, float]] = {
        "classification": (4.0, 0.0),
        "bbox": (3.0, 7.0),
        "polygon": (5.0, 25.0),
        "keypoints": (5.0, 18.0),
    }

    base_img_sec, per_obj_sec = timing_sec[annotation_type]
    sec_per_image = base_img_sec + (avg_objects_per_image * per_obj_sec)

    # Pre-labeling (YOLO/SAM) зменшує витрати часу на ручне обведення на 60%
    assist_multiplier = 0.40 if prelabeling_assist else 1.0
    labeling_hours = (num_images * sec_per_image * assist_multiplier) / 3600.0

    # QA-перевірка займає ~35% часу від повної ручної розмітки на відібрану частку кадрів
    qa_images = num_images * (qa_review_pct / 100.0)
    qa_hours = (qa_images * sec_per_image * 0.35) / 3600.0

    total_hours = labeling_hours + qa_hours
    total_cost_usd = total_hours * hourly_rate_usd
    cost_per_image_usd = total_cost_usd / num_images
    # 6 продуктивних годин активної розмітки на 1 анотатора за робочий день
    annotator_days = total_hours / 6.0

    payload = {
        "num_images": num_images,
        "annotation_type": annotation_type,
        "avg_objects_per_image": avg_objects_per_image,
        "prelabeling_assist": prelabeling_assist,
        "qa_review_pct": qa_review_pct,
        "hourly_rate_usd": hourly_rate_usd,
        "labeling_hours": round(labeling_hours, 2),
        "qa_hours": round(qa_hours, 2),
        "total_hours": round(total_hours, 2),
        "estimated_cost_usd": round(total_cost_usd, 2),
        "cost_per_image_usd": round(cost_per_image_usd, 4),
        "annotator_days_6h": round(annotator_days, 2),
    }
    return json.dumps(payload, ensure_ascii=False)


# Єдиний експортований реєстр інструментів для LangGraph
ALL_TOOLS = [
    estimate_gpu_vram,
    recommend_training_config,
    audit_annotation_batch,
    estimate_labeling_budget,
]


# ── Автономні unit-тести (Завдання 1: успішні виклики + перевірка валідації) ──
if __name__ == "__main__":
    print("=== 1. Тести валідації Pydantic (очікувані помилки) ===")
    invalid_cases = [
        ("Неіснуюча модель", lambda: EstimateVramInput(model_name="yolo99x")),
        ("img_size не кратний 32", lambda: EstimateVramInput(model_name="yolo11n", img_size=650)),
        ("int4 у режимі train", lambda: EstimateVramInput(model_name="yolo11n", mode="train", precision="int4")),
        ("reviewed > total", lambda: AuditAnnotationBatchInput(
            batch_id="BATCH-1", total_images=100, reviewed_images=120,
            mean_iou=0.8, honeypot_accuracy=0.9, missing_label_rate=0.02,
        )),
        ("Негативна ставка", lambda: EstimateLabelingBudgetInput(num_images=100, hourly_rate_usd=-5.0)),
    ]

    for label, fn in invalid_cases:
        try:
            fn()
            raise AssertionError(f"Очікувалась ValidationError для: {label}")
        except ValueError as err:
            msg = str(err).splitlines()[0] if str(err).splitlines() else str(err)
            print(f"[OK] {label} -> перехоплено: {msg}")

    print("\n=== 2. Тести валідних викликів усіх 4 tools ===")
    t1 = estimate_gpu_vram.invoke(
        {"model_name": "yolo11m", "mode": "train", "precision": "fp16", "batch_size": 16, "img_size": 640, "gpu_vram_gb": 6.0}
    )
    print("estimate_gpu_vram:", t1)

    t2 = recommend_training_config.invoke(
        {"model_name": "yolo11m", "gpu_vram_gb": 6.0, "img_size": 640, "target_effective_batch_size": 16}
    )
    print("recommend_training_config:", t2)

    t3 = audit_annotation_batch.invoke(
        {
            "batch_id": "batch-042",
            "total_images": 5000,
            "reviewed_images": 500,
            "mean_iou": 0.71,
            "honeypot_accuracy": 0.82,
            "missing_label_rate": 0.09,
            "strict_mode": True,
        }
    )
    print("audit_annotation_batch:", t3)

    t4 = estimate_labeling_budget.invoke(
        {
            "num_images": 5000,
            "avg_objects_per_image": 6.0,
            "annotation_type": "polygon",
            "prelabeling_assist": True,
            "qa_review_pct": 15.0,
            "hourly_rate_usd": 12.0,
        }
    )
    print("estimate_labeling_budget:", t4)
    print("\nУсі unit-тести Завдання 1 пройдено успішно.")