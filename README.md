# ReAct-агент для MLOps та Computer Vision

## 1. Опис проєкту
Побудовано автономного ReAct-агента на базі **LangGraph** для вирішення практичних інженерних задач MLOps та Computer Vision DataOps. Агент працює на **Google Gemini API (`gemini-3.1-flash-lite`)** через `langchain-google-genai`, використовує 4 доменні інструменти зі строгою валідацією **Pydantic v2** та повертає структуровану фінальну відповідь (`AgentResponse`: `answer`, `confidence`, `sources`). Реалізовано повний набір production-захистів (`max_steps`, загальний `timeout` на базі `time.monotonic()`, `LoopDetector` для виявлення зациклення, rate-limit backoff для HTTP 429/503) і покрокове JSON-логування траєкторії виконання (`trajectory.json`).

## 2. Доменна задача
- **Домен:** MLOps / Computer Vision Engineering & DataOps QA.
- **Що вирішує агент:** допомагає ML-інженерам і DataOps-командам оцінювати споживання GPU VRAM перед запуском тренування, підбирати безпечні конфігурації (`micro_batch_size`, `gradient_accumulation_steps`, `precision`) під апаратні ліміти відеокарти (наприклад, RTX 2060 6 GB), проводити автоматизований QA-аудит партій розмічених зображень та розраховувати бюджет і трудовитрати на розмітку або перерозмітку датасетів.
- **Інструменти (tools):**
  - `estimate_gpu_vram` — детально розраховує споживання VRAM (ваги, оптимізатор AdamW/градієнти, активації з квадратичним масштабуванням до `img_size`, CUDA overhead), утилізацію у % та ризик OOM (`SAFE`, `TIGHT_HEADROOM`, `HIGH_OOM_RISK`, `OOM_EXCEEDED`).
  - `recommend_training_config` — підбирає безпечний `micro_batch_size` і `gradient_accumulation_steps` для досягнення цільового ефективного батчу в межах <=85% доступної VRAM.
  - `audit_annotation_batch` — виконує QA-аудит партії розмітки за `mean_iou`, `honeypot_accuracy`, `missing_label_rate` та покриттям перевірки у стандартному або строгому режимі; повертає вердикт (`APPROVED`, `CONDITIONAL_REWORK`, `REJECTED`) та обсяг перерозмітки.
  - `estimate_labeling_budget` — оцінює години роботи (розмітка + QA), загальний бюджет у USD, ціну за зображення та людино-дні з урахуванням типу розмітки (`classification`, `bbox`, `polygon`, `keypoints`) і використання semi-automated pre-labeling (YOLO/SAM).

## 3. Архітектура
- **LLM-провайдер:** Google Gemini API (`gemini-3.1-flash-lite`, підключення через `ChatGoogleGenerativeAI`; для моделей серії Gemini 3 збережено дефолтну температуру `1.0` для запобігання зацикленню).
- **Структура графа LangGraph (`StateGraph`):**
```text
START ──► agent (LLM + bind_tools + SafetyGuard)
             ├── [є tool_calls] ──► tools (ToolNode з handle_tool_errors=True) ──► agent
             └── [немає tool_calls або safety stop] ──► formatter (with_structured_output) ──► END
```
- **Structured Outputs:** вузол `formatter` формує Pydantic-модель `AgentResponse(answer, confidence, sources)` на основі фактичних результатів `ToolMessage` (Ground Truth grounding) і має безпечний fallback у разі збою або перевищення rate limit.
- **Захисні механізми (`safety.py`):**
  - `max_steps = 10` — жорсткий ліміт ітерацій ReAct-циклу з поверненням часткової відповіді.
  - `timeout = 120.0` сек — контроль загального часу виконання через `time.monotonic()` із fail-fast зупинкою, якщо очікування rate limit перевищує залишок бюджету часу.
  - `LoopDetector (max_repeats = 3)` — MD5-хешування канонічної пари `(tool_name, args)`; зупиняє граф при 3 однакових викликах поспіль.
  - Обробка транзієнтних помилок API (`429 RESOURCE_EXHAUSTED`, `503 UNAVAILABLE`) із паузами в межах загального тайм-ауту.
- **Спостережуваність (`logger.py`):** `TrajectoryLogger` фіксує кожен вузол (`step_number`, `node_name`, `input`, `output`, `tool_calls`, `timestamp`, `duration_ms`, `elapsed_ms`) із обрізанням довгих рядків до 500 символів та збереженням у `trajectory.json` (`ensure_ascii=False`, `indent=2`).

## 4. Інструкція запуску
1. Клонувати репозиторій:
```bash
git clone <repo-url>
cd hw1_react_agent
```
2. Створити та активувати віртуальне середовище:
```bash
python3 -m venv .venv_hw1
source .venv_hw1/bin/activate
```
3. Встановити залежності:
```bash
pip install -r requirements.txt
```
4. Налаштувати змінні оточення: скопіювати `.env.example` у `.env` і вказати власний ключ Google AI Studio:
```bash
cp .env.example .env
# відредагуйте .env та вставте GOOGLE_API_KEY
```
5. Запустити перевірки поетапно:
   - Unit-тести Pydantic-схем та інструментів (Завдання 1): `python tools.py`
   - Unit-тести `LoopDetector`, `max_steps` і `timeout`: `python safety.py`
   - Демонстрація захисних механізмів у графі + реальний запит (Завдання 2–3): `python agent.py`
   - Повний прогін 5 тест-кейсів із генерацією `test_results.json` та `trajectory.json` (Завдання 4):
```bash
python test_runner.py
```

## 5. Результати тестування
Підсумкова таблиця за фінальним прогоном `test_results.json` (модель `gemini-3.1-flash-lite`, `max_steps=10`, `timeout=120s`, `max_repeats=3`):

| Test ID | Складність | Кроків | Tool calls | Час, мс | Статус |
|---------|------------|--------|------------|---------|--------|
| TC-001 | simple | 2 | `estimate_gpu_vram` x 1 | 4870 | ✅ success |
| TC-002 | medium | 2 | `estimate_gpu_vram` x 2 | 5636 | ✅ success |
| TC-003 | medium | 2 | `estimate_gpu_vram` x 1, `recommend_training_config` x 1 | 5191 | ✅ success |
| TC-004 | complex | 4 | `audit_annotation_batch` x 1, `estimate_labeling_budget` x 2 | 24013 | ✅ success |
| TC-005 | complex | 2 | `estimate_gpu_vram` x 1, `recommend_training_config` x 1, `estimate_labeling_budget` x 1 | 5210 | ✅ success |

*Примітка: у `TC-003` час включає 13 с паузи очікування після HTTP 429 (~13 с чистого виконання); у `TC-005` час включає 90 с (15 + 30 + 45 с) backoff-спроб у вузлі `formatter` через Free Tier RPM, після чого спрацював безпечний fallback (~15 с чистого виконання).*

## 6. Аналіз результатів

**Чи правильно агент обирав tools?**
У фінальному прогоні агент обрав правильні інструменти у 5 із 5 тест-кейсів (100% точність маршрутизації). У `TC-001` та `TC-002` коректно використано `estimate_gpu_vram` (включно з паралельним викликом для `fp32` і `fp16`). У `TC-003` модель одночасно викликала оцінку VRAM для `batch_size=8` (підтвердивши `OOM_EXCEEDED`: 7.45 GB > 6.0 GB) та `recommend_training_config`. У `TC-004` агент проявив справжнє багатокрокове планування: спочатку викликав `audit_annotation_batch`, отримав вердикт `REJECTED` (4000 кадрів на rework), і лише на 2-му кроці викликав `estimate_labeling_budget` двічі — із `prelabeling_assist=False` та `prelabeling_assist=True`. У `TC-005` агент коректно зчитав зі схеми `EstimateVramInput`, що `int4` допустимий лише для `inference`, явно пояснив це обмеження користувачу й паралельно виконав усі три потрібні розрахунки.

**Де агент помилявся і чому?**
Під час ітеративного налагодження за допомогою `trajectory.json` було виявлено та усунено два типові failure modes LLM-агентів:
1. *Галюцинації у вузлі `formatter` при порожній чернетці (до фіксу):* під час проміжного тестування на `gemini-2.5-flash-lite` модель на кроці `[3]` повертала порожній `content` (`""`) після `ToolMessage`. Оскільки `formatter` отримував лише текст останнього повідомлення без сирих `ToolMessage`, він вигадував числа з голови (наприклад, `7.5 GB` замість реальних `1.68 GB` у `TC-001` та `4.5 GB` замість `7.45 GB` у `TC-003`). Проблему усунено передачею усіх `ToolMessage` у `formatter_node` як явного блоку `GROUND TRUTH`.
2. *Зовнішні обмеження API (`429 RESOURCE_EXHAUSTED` та `503 UNAVAILABLE`):* на Free Tier інтенсивні тести періодично отримували rate limit або тимчасову недоступність моделі. Додавання `invoke_with_rate_limit_retry` із fail-fast перевіркою залишку `timeout` та fallback у `formatter` дозволило уникнути падінь у `TC-003` і `TC-005`.

**Кроки: simple vs complex.**
Кількість кроків LLM виявилася майже сталою, а не лінійною відносно кількості підзадач: `simple` (`TC-001`) — 2 кроки, `medium` (`TC-002`, `TC-003`) — 2 кроки, `complex` (`TC-005`) — 2 кроки, і лише `TC-004` потребував 3 кроків. Причина — **паралельний Tool Calling**: коли підзадачі незалежні (`TC-002`, `TC-003`, `TC-005`), Gemini викликає 2–3 інструменти за одну ітерацію. Натомість у `TC-004` присутня логічна залежність (*«якщо партію відхилять — порахуй перерозмітку»*), тому агент виконав залежні інструменти послідовно за 3 кроки.

**Чи досягав агент max_steps або timeout?**
У фінальному прогоні жоден тест-кейс не досяг `max_steps=10` (максимум становив 3 кроки — запас 70%), `timeout=120s` (найдовший тест `TC-005` завершився за 105.9 с з урахуванням 90 с backoff) або `LoopDetector`. Окремо всі три захисні механізми верифіковано та продемонстровано в `agent.py` (на детермінованій мок-LLM: `stop_reason=max_steps` на кроці 3, `stop_reason=loop_detected` при 3 однакових викликах, `stop_reason=timeout` при перевищенні 0.4 с) та в `safety.py`.

**Залежність часу виконання від складності.**
Власний час виконання Python-інструментів становить лише **1–3 мс**, тому загальна затримка (latency) майже на 100% визначається мережевими викликами LLM та політиками rate limit. Без урахування пауз 429 спостерігається помірне зростання чистої латентності від ~10.7 с (`simple`, 2 виклики LLM + formatter) до ~21.7 с (`complex` з послідовними кроками `TC-004`, 3 виклики LLM + formatter).

**Випадки, коли агент НЕ використав tool, хоч мав би.**
На ранній версії системного промпту під час тестового запуску `TC-004` на `gemini-2.5-flash` агент коректно викликав `audit_annotation_batch`, але **проігнорував `estimate_labeling_budget`** і самостійно порахував бюджет "в голові" (`200 год / $2200` замість реальних `69.0 год / $758.97`), при цьому вказавши `confidence: 1.0`. Після посилення `SYSTEM_PROMPT` із прямою таблицею відповідності задач інструментам і забороною ручних підрахунків проблема повністю зникла у фінальних прогонах.

## 7. Висновки та обмеження
- **Що працює добре:**
  - Контракти Pydantic v2 із `Field`, `field_validator` та `model_validator` надійно відсікають некоректні аргументи (`img_size` не кратний 32, `int4` у `train`, `reviewed > total`) до виконання логіки.
  - Заземлення вузла `formatter` на фактичні `ToolMessage` гарантує відсутність числових галюцинацій у фінальному `AgentResponse`.
  - Комбінація `max_steps`, `timeout`, `LoopDetector` і rate-limit retry забезпечує передбачувану поведінку графа навіть при збоях зовнішнього API.
- **Що можна покращити:**
  - Поєднати останній крок `agent_node` та `formatter_node` (або використовувати `response_format` на рівні графа), щоб скоротити кількість LLM-викликів на 1 запит за прогін і знизити latency на ~30%.
  - Підключити реальний моніторинг GPU (`pynvml`/`nvidia-smi`) та експорт метрик розмітки з Label Studio / CVAT API замість параметричних формул.
  - Додати автоматичні eval-метрики (`deepeval` — `Faithfulness` і `ToolCorrectness`) у CI-пайплайн.
- **Відомі обмеження:**
  - Формули оцінки VRAM є аналітичною апроксимацією і можуть відрізнятися на +-10–15% залежно від версії CUDA, PyTorch allocator та аугментацій.
  - На Gemini Free Tier жорсткі обмеження RPM/RPD збільшують час прогону серії тестів через необхідність пауз між запитами.

## 8. Структура файлів
- `tools.py` — Pydantic v2 схеми, валідатори, 4 доменні tools та unit-тести
- `agent.py` — LangGraph граф (`agent` <-> `tools` -> `formatter`), `AgentResponse`, демонстрація захисних механізмів
- `safety.py` — `SafetyConfig`, `SafetyGuard` (`max_steps`, `timeout`, 429/503 retry) та `LoopDetector`
- `logger.py` — `TrajectoryLogger` для збереження повної траєкторії у JSON
- `test_runner.py` — запуск 5 тест-кейсів різної складності та збір метрик
- `trajectory.json` — повний лог траєкторії виконання (згенеровано)
- `test_results.json` — результати 5 тест-кейсів (згенеровано)
- `requirements.txt` — залежності проєкту
- `.env.example` — шаблон змінних оточення (без секретів)
- `README.md` — документація, інструкція запуску та аналіз результатів
