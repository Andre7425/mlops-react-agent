"""
LangGraph ReAct-агент для MLOps / Computer Vision із:
- циклом LLM–tools–LLM,
- Pydantic v2 Structured Outputs,
- захисними механізмами (max_steps, timeout, LoopDetector, 429 backoff),
- повним JSON-логуванням траєкторії виконання.
"""

from __future__ import annotations

import json
import os
import time
from typing import Annotated, Any, Literal, Optional, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field

from logger import TrajectoryLogger
from safety import SafetyConfig, SafetyGuard
from tools import ALL_TOOLS

# Завантажуємо змінні оточення з .env поруч із файлом
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

SYSTEM_PROMPT = """Ти — досвідчений MLOps та Computer Vision інженер-асистент.
Твоє завдання — надавати точні технічні розрахунки щодо GPU VRAM, конфігурацій навчання,
аудиту якості розмітки (DataOps QA) та бюджетування датасетів.

Обов'язкові правила роботи:
1. ЗАБОРОНЕНО самостійно рахувати в голові споживання VRAM, batch_size, вердикт QA-аудиту,
   години або вартість розмітки. Для КОЖНОГО такого підрахунку ти ЗОБОВ'ЯЗАНИЙ викликати
   відповідний інструмент:
   - VRAM -> estimate_gpu_vram
   - безпечний batch/grad_accum -> recommend_training_config
   - QA-аудит партії -> audit_annotation_batch
   - години та бюджет розмітки/перерозмітки -> estimate_labeling_budget
   Якщо користувач просить порівняти 2 варіанти (наприклад, з pre-labeling і без нього,
   або fp32 проти fp16) — викликай відповідний інструмент окремо для КОЖНОГО варіанту.
2. Якщо задача складна і складається з кількох частин — викликай відповідні
   інструменти послідовно або паралельно.
3. Якщо інструмент повернув помилку валідації, проаналізуй повідомлення про помилку,
   скоригуй параметри та повтори виклик.
4. Формуй чітку, структуровану фінальну відповідь українською мовою з конкретними цифрами
   та практичними рекомендаціями на основі результатів інструментів.
"""


# ── Pydantic-модель для Structured Output ─────────────────────────────────────

class AgentResponse(BaseModel):
    """Структурована фінальна відповідь ReAct-агента."""

    answer: str = Field(
        description="Повна, чітка технічна відповідь користувачу українською мовою з конкретними цифрами."
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Оцінка впевненості у відповіді від 0.0 до 1.0 (1.0, якщо всі цифри підтверджені tools).",
    )
    sources: list[str] = Field(
        default_factory=list,
        description="Список назв інструментів (tools), дані яких використані для формування відповіді.",
    )


# ── Стан графа LangGraph ──────────────────────────────────────────────────────

class AgentState(TypedDict):
    """Стан, що передається між вузлами графа."""

    messages: Annotated[list[BaseMessage], add_messages]
    step_count: int
    stop_reason: Optional[str]
    structured_response: Optional[AgentResponse]


# ── Допоміжні функції ─────────────────────────────────────────────────────────

def extract_text(content: Any) -> str:
    """Безпечно витягує текст із content (підтримує str та list[dict] від Gemini)."""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return "\n".join(parts).strip()
    return str(content).strip()


def extract_used_tools(messages: list[BaseMessage]) -> list[str]:
    """Повертає впорядкований список унікальних інструментів, викликаних у сесії."""
    seen: list[str] = []
    for msg in messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for call in msg.tool_calls:
                name = call.get("name")
                if name and name not in seen:
                    seen.append(name)
    return seen


def extract_all_tool_calls(messages: list[BaseMessage]) -> list[str]:
    """Повертає повний хронологічний список усіх викликів інструментів (із дублікатами)."""
    calls: list[str] = []
    for msg in messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for call in msg.tool_calls:
                calls.append(str(call.get("name", "unknown")))
    return calls


def extract_tool_observations(messages: list[BaseMessage]) -> str:
    """Збирає всі фактичні відповіді інструментів (ToolMessage) як Ground Truth."""
    obs: list[str] = []
    for idx, msg in enumerate(messages, start=1):
        if isinstance(msg, ToolMessage):
            name = getattr(msg, "name", "tool")
            obs.append(f"{idx}. [{name}]: {extract_text(msg.content)}")
    return "\n".join(obs) if obs else "(інструменти не викликались)" 


def create_llm(model_name: Optional[str] = None) -> ChatGoogleGenerativeAI:
    """Ініціалізує Google Gemini LLM із перевіркою API-ключа."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "Не знайдено GOOGLE_API_KEY. Додайте ключ у файл .env або експортуйте його в середовище."
        )

    selected_model = (model_name or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")).strip()

    kwargs: dict[str, Any] = {
        "model": selected_model,
        "google_api_key": api_key,
        "timeout": 40,
        "max_retries": 0,  # Повторами 429 керує SafetyGuard у межах загального timeout
    }
    # Для моделей Gemini 3.x залишаємо дефолтну температуру (1.0)
    if not selected_model.startswith("gemini-3"):
        kwargs["temperature"] = 0.1

    return ChatGoogleGenerativeAI(**kwargs)


# ── Побудова захищеного графа LangGraph ───────────────────────────────────────

def build_agent_graph(
    llm: Optional[Any] = None,
    safety_config: Optional[SafetyConfig] = None,
    trajectory_logger: Optional[TrajectoryLogger] = None,
    verbose: bool = True,
):
    """Збирає LangGraph StateGraph із захистами (max_steps, timeout, LoopDetector) і логером."""
    base_llm = llm or create_llm()
    llm_with_tools = base_llm.bind_tools(ALL_TOOLS)
    structured_llm = base_llm.with_structured_output(AgentResponse)

    guard = SafetyGuard(safety_config or SafetyConfig())
    logger = trajectory_logger or TrajectoryLogger()
    raw_tool_node = ToolNode(ALL_TOOLS, handle_tool_errors=True)
    log_step_counter = {"n": 0}

    def next_log_step() -> int:
        log_step_counter["n"] += 1
        return log_step_counter["n"]

    # ── Вузол 1: agent (з перевірками timeout, max_steps, LoopDetector) ──
    def agent_node(state: AgentState) -> dict:
        t0 = time.monotonic()
        current_step = state.get("step_count", 0)
        last_input = extract_text(state["messages"][-1].content) if state["messages"] else ""

        # 1. Перевірка timeout до виклику LLM
        if guard.is_timed_out():
            msg_text = (
                f"Перевищено загальний тайм-аут виконання ({guard.config.timeout_seconds:.1f} с). "
                f"Виконання зупинено після {current_step} кроків."
            )
            if verbose:
                print(f"  [safety/timeout] {msg_text}", flush=True)
            logger.log_step(
                step_number=next_log_step(),
                node_name="agent_safety_stop",
                input_data=last_input,
                output_data=msg_text,
                duration_ms=int((time.monotonic() - t0) * 1000),
            )
            return {
                "messages": [AIMessage(content=msg_text)],
                "step_count": current_step,
                "stop_reason": "timeout",
            }

        # 2. Перевірка max_steps до виклику LLM
        if guard.is_max_steps_reached(current_step):
            msg_text = (
                f"Досягнуто ліміт кроків ({guard.config.max_steps}). "
                f"Часткова відповідь: оброблено {current_step} ітерацій ReAct-циклу."
            )
            if verbose:
                print(f"  [safety/max_steps] {msg_text}", flush=True)
            logger.log_step(
                step_number=next_log_step(),
                node_name="agent_safety_stop",
                input_data=last_input,
                output_data=msg_text,
                duration_ms=int((time.monotonic() - t0) * 1000),
            )
            return {
                "messages": [AIMessage(content=msg_text)],
                "step_count": current_step,
                "stop_reason": "max_steps",
            }

        # 3. Виклик LLM (з контролем rate-limit 429)
        next_step = current_step + 1
        prompt_messages = [SystemMessage(content=SYSTEM_PROMPT), *state["messages"]]
        if verbose:
            print(f"  [agent] крок {next_step}: запит до LLM...", flush=True)

        try:
            response: AIMessage = guard.invoke_with_rate_limit_retry(
                lambda: llm_with_tools.invoke(prompt_messages)
            )
        except TimeoutError as exc:
            msg_text = str(exc)
            if verbose:
                print(f"  [safety/timeout] {msg_text}", flush=True)
            logger.log_step(
                step_number=next_log_step(),
                node_name="agent_safety_stop",
                input_data=last_input,
                output_data=msg_text,
                duration_ms=int((time.monotonic() - t0) * 1000),
            )
            return {
                "messages": [AIMessage(content=msg_text)],
                "step_count": next_step,
                "stop_reason": "timeout",
            }

        # 4. Перевірка timeout після відповіді LLM (захист від занадто повільного виклику)
        if guard.is_timed_out():
            msg_text = (
                f"Перевищено загальний тайм-аут ({guard.config.timeout_seconds:.1f} с) "
                f"під час кроку {next_step}."
            )
            if verbose:
                print(f"  [safety/timeout] {msg_text}", flush=True)
            logger.log_step(
                step_number=next_log_step(),
                node_name="agent_safety_stop",
                input_data=last_input,
                output_data=msg_text,
                duration_ms=int((time.monotonic() - t0) * 1000),
            )
            return {
                "messages": [AIMessage(content=msg_text)],
                "step_count": next_step,
                "stop_reason": "timeout",
            }

        tool_calls = response.tool_calls or []
        tool_names = [str(c.get("name", "unknown")) for c in tool_calls]

        # 5. Детекція зациклення (повторюваних tool calls)
        if tool_calls:
            loop_sig = guard.check_tool_calls_for_loop(tool_calls)
            if loop_sig:
                msg_text = (
                    f"Виявлено зациклення (LoopDetector): інструмент {loop_sig} "
                    f"викликався {guard.config.max_repeats} рази поспіль. Виконання зупинено."
                )
                if verbose:
                    print(f"  [safety/loop] {msg_text}", flush=True)
                logger.log_step(
                    step_number=next_log_step(),
                    node_name="agent_safety_stop",
                    input_data=last_input,
                    output_data=msg_text,
                    tool_calls=tool_names,
                    duration_ms=int((time.monotonic() - t0) * 1000),
                )
                return {
                    "messages": [AIMessage(content=msg_text)],
                    "step_count": next_step,
                    "stop_reason": "loop_detected",
                }

        duration_ms = int((time.monotonic() - t0) * 1000)
        if verbose:
            print(f"  [agent] крок {next_step}: tool_calls={tool_names} ({duration_ms} мс)", flush=True)

        logger.log_step(
            step_number=next_log_step(),
            node_name="agent",
            input_data=last_input,
            output_data=extract_text(response.content) or (f"tool_calls: {tool_names}" if tool_names else "(empty response)"),
            tool_calls=tool_names,
            duration_ms=duration_ms,
        )
        return {
            "messages": [response],
            "step_count": next_step,
            "stop_reason": None,
        }

    # ── Вузол 2: tools (обгортка навколо ToolNode для логування кожного виклику) ──
    def tools_node(state: AgentState) -> dict:
        t0 = time.monotonic()
        last_ai = state["messages"][-1]
        requested_calls = last_ai.tool_calls if isinstance(last_ai, AIMessage) else []
        call_summary = [
            {"name": c.get("name"), "args": c.get("args", {})} for c in requested_calls
        ]

        tool_result = raw_tool_node.invoke(state)
        tool_messages: list[BaseMessage] = tool_result.get("messages", [])
        outputs_preview = [
            f"{getattr(m, 'name', 'tool')}: {extract_text(m.content)}"
            for m in tool_messages
            if isinstance(m, ToolMessage)
        ]

        duration_ms = int((time.monotonic() - t0) * 1000)
        if verbose:
            names = [c["name"] for c in call_summary]
            print(f"  [tools] виконано {names} ({duration_ms} мс)", flush=True)

        logger.log_step(
            step_number=next_log_step(),
            node_name="tools",
            input_data=call_summary,
            output_data=" | ".join(outputs_preview),
            tool_calls=[str(c["name"]) for c in call_summary],
            duration_ms=duration_ms,
        )
        return tool_result

    # ── Вузол 3: formatter (Pydantic Structured Output) ──
    def formatter_node(state: AgentState) -> dict:
        t0 = time.monotonic()
        last_text = extract_text(state["messages"][-1].content)
        used_tools = extract_used_tools(state["messages"])
        stop_reason = state.get("stop_reason")

        # Якщо спрацював захисний механізм — формуємо структуровану відповідь локально без виклику LLM
        if stop_reason in ("max_steps", "timeout", "loop_detected"):
            structured = AgentResponse(
                answer=last_text,
                confidence=0.3 if used_tools else 0.1,
                sources=used_tools,
            )
            logger.log_step(
                step_number=next_log_step(),
                node_name="formatter_fallback",
                input_data=f"stop_reason={stop_reason}",
                output_data=structured.model_dump(),
                tool_calls=used_tools,
                duration_ms=int((time.monotonic() - t0) * 1000),
            )
            return {"structured_response": structured}

        user_queries = [
            extract_text(m.content) for m in state["messages"] if isinstance(m, HumanMessage)
        ]
        original_query = user_queries[0] if user_queries else ""

        tool_observations = extract_tool_observations(state["messages"])

        formatting_prompt = (
            "Сформуй структуровану фінальну відповідь MLOps-агента у форматі AgentResponse.\n"
            f"Запит користувача: {original_query}\n"
            f"Фактично викликані інструменти: {used_tools}\n\n"
            f"Фактичні результати інструментів (GROUND TRUTH — єдине джерело цифр):\n{tool_observations}\n\n"
            f"Чернетка відповіді агента (може бути порожньою):\n{last_text or '(порожньо)'}\n\n"
            "Суворі вимоги:\n"
            "- У поле 'answer' запиши повну, чітку технічну відповідь українською мовою, "
            "спираючись ВИКЛЮЧНО на цифри й статуси з блоку GROUND TRUTH. Заборонено вигадувати або змінювати числа.\n"
            "- Якщо один із викликів повернув помилку валідації (наприклад, заборона int4 для train), "
            "явно поясни це користувачу й наведи дані з успішних викликів.\n"
            "- У поле 'sources' включи лише ті інструменти, які реально викликалися.\n"
            "- У поле 'confidence' постав оцінку від 0.0 до 1.0."
        )

        try:
            structured: AgentResponse = guard.invoke_with_rate_limit_retry(
                lambda: structured_llm.invoke([HumanMessage(content=formatting_prompt)])
            )
            if not structured.sources and used_tools:
                structured = structured.model_copy(update={"sources": used_tools})
        except Exception as exc:
            if verbose:
                print(f"  [formatter] fallback через помилку ({type(exc).__name__})", flush=True)
            fallback_answer = last_text or f"Результати інструментів:\n{tool_observations}"
            structured = AgentResponse(
                answer=fallback_answer,
                confidence=0.85 if used_tools else 0.5,
                sources=used_tools,
            )

        duration_ms = int((time.monotonic() - t0) * 1000)
        logger.log_step(
            step_number=next_log_step(),
            node_name="formatter",
            input_data=last_text or f"(empty draft; grounded on {len(used_tools)} tool outputs)",
            output_data=structured.model_dump(),
            tool_calls=used_tools,
            duration_ms=duration_ms,
        )
        return {"structured_response": structured}

    # ── Router ──
    def should_continue(state: AgentState) -> Literal["tools", "formatter"]:
        if state.get("stop_reason"):
            return "formatter"
        last_msg = state["messages"][-1]
        if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
            return "tools"
        return "formatter"

    # Збірка графа
    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tools_node)
    graph.add_node("formatter", formatter_node)

    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", "formatter": "formatter"},
    )
    graph.add_edge("tools", "agent")
    graph.add_edge("formatter", END)

    compiled = graph.compile()
    return compiled, guard, logger


# ── Демонстрація захисних механізмів (Завдання 3) + реальний запит Gemini ─────

class _MockBoundLLM:
    """Мок-LLM для детермінованої перевірки max_steps, timeout та LoopDetector."""

    def __init__(self, mode: Literal["loop", "max_steps", "timeout"]) -> None:
        self.mode = mode
        self.counter = 0

    def bind_tools(self, tools: Any) -> "_MockBoundLLM":
        return self

    def with_structured_output(self, schema: Any) -> "_MockBoundLLM":
        return self

    def invoke(self, messages: Any) -> AIMessage:
        self.counter += 1
        if self.mode == "timeout":
            time.sleep(0.6)
            return AIMessage(content="Занадто повільна відповідь")

        if self.mode == "loop":
            # Навмисно повторюємо однаковий tool call із тими самими аргументами
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": f"call_loop_{self.counter}",
                        "name": "estimate_gpu_vram",
                        "args": {"model_name": "yolo11m", "batch_size": 16, "gpu_vram_gb": 6.0},
                    }
                ],
            )

        # mode == "max_steps": щоразу інші аргументи (LoopDetector не спрацює, спрацює max_steps)
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "id": f"call_step_{self.counter}",
                    "name": "estimate_gpu_vram",
                    "args": {"model_name": "yolo11m", "batch_size": self.counter, "gpu_vram_gb": 6.0},
                }
            ],
        )


def demonstrate_safety_mechanisms() -> None:
    """Демонструє спрацювання всіх 3 захисних механізмів усередині LangGraph."""
    scenarios = [
        ("1. Демонстрація max_steps=3", "max_steps", SafetyConfig(max_steps=3, timeout_seconds=30, max_repeats=3)),
        ("2. Демонстрація LoopDetector (max_repeats=3)", "loop", SafetyConfig(max_steps=10, timeout_seconds=30, max_repeats=3)),
        ("3. Демонстрація timeout=0.4с", "timeout", SafetyConfig(max_steps=10, timeout_seconds=0.4, max_repeats=3)),
    ]

    for title, mode, cfg in scenarios:
        print(f"\n{'=' * 75}\n{title}\n{'=' * 75}")
        mock_llm = _MockBoundLLM(mode=mode)
        app, _, _ = build_agent_graph(llm=mock_llm, safety_config=cfg, verbose=True)
        res = app.invoke(
            {
                "messages": [HumanMessage(content=f"Тест механізму {mode}")],
                "step_count": 0,
                "stop_reason": None,
                "structured_response": None,
            }
        )
        print(f"stop_reason: {res.get('stop_reason')} | step_count: {res.get('step_count')}")
        if res.get("structured_response"):
            print("answer:", res["structured_response"].answer)


if __name__ == "__main__":
    # Частина А: швидка демонстрація всіх 3 захисних механізмів
    demonstrate_safety_mechanisms()

    # Частина Б: реальний складний DataOps-запит на Gemini зі збереженням trajectory.json
    print(f"\n{'=' * 75}\n4. Реальний запит Gemini + TrajectoryLogger -> trajectory.json\n{'=' * 75}")
    query = (
        "Проведи строгий QA-аудит партії DRONE-07: 4000 зображень, перевірено 450, "
        "mean_iou=0.72, honeypot_accuracy=0.84, missing_label_rate=0.06. "
        "Якщо партію відхилять, порахуй бюджет повної перерозмітки для bbox "
        "(середньо 8 об'єктів/кадр, зі ставкою $11/год, 15% QA) із pre-labeling та без нього."
    )
    traj_logger = TrajectoryLogger()
    traj_logger.reset(metadata={"demo": "real_gemini_dataops", "query": query})

    app, _, _ = build_agent_graph(
        safety_config=SafetyConfig(max_steps=10, timeout_seconds=120.0, max_repeats=3),
        trajectory_logger=traj_logger,
        verbose=True,
    )
    result = app.invoke(
        {
            "messages": [HumanMessage(content=query)],
            "step_count": 0,
            "stop_reason": None,
            "structured_response": None,
        }
    )

    traj_logger.save("trajectory.json")
    structured: Optional[AgentResponse] = result.get("structured_response")
    print(f"\nstop_reason: {result.get('stop_reason')} | кроків LLM: {result.get('step_count')}")
    if structured:
        print(json.dumps(structured.model_dump(), ensure_ascii=False, indent=2))