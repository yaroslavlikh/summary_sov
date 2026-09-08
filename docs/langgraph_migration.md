# Миграция на LangGraph + полный Langfuse-трейсинг

## Зачем

Три структурные проблемы, найденные за сегодня, решаются одним архитектурным сдвигом, а не серией патчей:

1. **Нет явного intent-классификатора** — rewrite-промпт решает "дейксис это или self-identity" одним недетерминированным вызовом с конкурирующими правилами (баг "о чём это я"). LangGraph заставляет сделать классификацию отдельным узлом графа с явным условным переходом — по определению нет способа "забыть" порядок приоритета, потому что порядок — это структура графа, а не текст промпта.
2. **`is_bot` фильтруется непоследовательно** — потому что SQL-запросы разбросаны по функциям. Если retrieval — это узлы графа с одним источником правды на функцию, фильтр пишется один раз в одном месте.
3. **Ручное логирование через `@observe`/`update_current_span`** — нужно не забыть добавить в каждую новую функцию (та же болезнь, что и `is_bot`). LangGraph + Langfuse даёт трейсинг «по умолчанию» на весь граф одним `.with_config()`, без единого забытого места.

## Новые зависимости

```
langgraph
langchain-core
langchain-groq
```

`langchain_groq.ChatGroq` — обёртка над Groq API в терминах LangChain `BaseChatModel`. Меняем на неё вызовы модели (не сырой `groq.Groq` клиент) — это то, что даёт Langfuse возможность трейсить **каждый** LLM-вызов автоматически через callback, без ручных `@observe` на каждую новую функцию.

## Langfuse — подключение (одна точка, не per-function)

```python
from langfuse.langchain import CallbackHandler

langfuse_handler = CallbackHandler()  # читает LANGFUSE_* из env, как и сейчас

# при вызове любого графа:
result = compiled_graph.invoke(initial_state, config={"callbacks": [langfuse_handler]})
```

Всё дерево вызовов внутри графа (каждый узел, каждый LLM-вызов внутри узла, включая tool calls) попадает в Langfuse одним деревом — ручные `langfuse.update_current_span(...)` по всему `handlers.py` можно выкинуть.

## Граф №1 — `/ask` (замена `answer_chat_question`)

### State

```python
from typing import TypedDict, Optional

class AskState(TypedDict):
    chat_id: int
    question: str                # оригинальный вопрос
    asker_name: str
    replied_message_id: Optional[int]
    bot_username: str
    thread_id: Optional[int]

    anchor_id: Optional[int]
    intent: str                  # "self_identity" | "deictic" | "self_contained" | "other_person"
    effective_question: str

    fts_ids: list[int]
    vector_ids: list[int]
    candidate_ids: list[int]
    match_ids: set[int]
    window_rows: list[tuple]

    answer: Optional[str]
```

### Узлы и переходы

```
START
  → resolve_anchor
      ├─ (anchor найден) → generate_answer
      └─ (anchor нет)     → classify_intent          # НОВЫЙ узел — тот самый роутер
             classify_intent → rewrite_query          # rewrite уже ЗНАЕТ intent, не гадает
             rewrite_query → [search_fts, search_vector]   # fan-out, параллельно
             [search_fts, search_vector] → fuse_rrf        # fan-in
             fuse_rrf → rerank
             rerank → generate_answer
  generate_answer → save_bot_answer → END
```

`classify_intent` — отдельный узел, ОДНОЗНАЧНО решающий категорию до всякого rewrite (устраняет конфликт правил внутри одного промпта):

```python
def classify_intent(state: AskState) -> AskState:
    result = intent_classifier_llm.invoke(
        f"Вопрос: {state['question']}\nКатегория (self_identity/deictic/self_contained/other_person)?"
    )
    state["intent"] = result.content.strip()
    return state

def route_by_intent(state: AskState) -> str:
    return "rewrite_query"  # rewrite сам решает КАК резолвить по intent, не решает ЧТО это за случай
```

`search_fts`/`search_vector` — здесь и только здесь пишется `AND is_bot = FALSE` (единственное место в кодовой базе):

```python
def search_fts(state: AskState) -> AskState:
    # SELECT ... WHERE user_id = %s AND is_bot = FALSE AND search_vector @@ ...
    ...
```

## Граф №2 — `/summary` + инкрементальный tool-calling

Самый конкретный выигрыш: наш ручной while-цикл в `extract_context_updates` (с багом зацикливания, который чинили дедупом) заменяется встроенным паттерном LangGraph:

```python
from langgraph.prebuilt import ToolNode, tools_condition

tool_node = ToolNode([update_portrait, add_chat_lore, record_moment])

graph.add_node("extract_context", context_extraction_llm_node)
graph.add_node("tools", tool_node)
graph.add_conditional_edges("extract_context", tools_condition)  # сам решает: ещё tool call или конец
graph.add_edge("tools", "extract_context")
```

`tools_condition` — стандартная, протестированная библиотекой функция: смотрит, есть ли `tool_calls` в последнем сообщении модели, и либо идёт в `tools`, либо в `END`. Наш баг с повторением одного и того же вызова был именно в том, что мы сами написали эту проверку руками и не учли все случаи — здесь это чужой, обкатанный код.

```
START → generate_summary → send_summary → extract_context ⇄ tools → END
```

## Фазы миграции (не всё сразу)

1. **Фаза 1** — только граф `/ask`, самый багатый узел сегодня. `/summary`, `/learncontext`, ingestion голоса/фото остаются как есть (обычные функции).
2. **Фаза 2** — граф `/summary` + `ToolNode` для extraction (убирает наш самописный цикл).
3. **Фаза 3** (опционально) — `/learncontext` как граф, если понадобится более сложная логика (сейчас там линейный map-reduce, графу нечего добавить, кроме трейсинга).

Не переносим на LangGraph: приём голоса/фото/стикеров (это чистый I/O + один вызов модели, граф не добавляет ценности), шифрование, миграции схемы — это инфраструктура, не агентная логика.

## Что НЕ меняется

- Postgres/pgvector, шифрование, схема БД — без изменений.
- `faster-whisper` для голоса — остаётся отдельным вызовом (LangChain не добавляет ценности к чистому STT).
- Vision-captioning картинок — можно оставить как прямой вызов `ChatGroq` с картинкой в content, не обязательно узел графа, если не встраивается в conditional-логику.

## Источники

- [Langfuse LangGraph integration](https://langfuse.com/guides/cookbook/integration_langgraph) — `CallbackHandler` + `.with_config()`.
- [Langfuse LangChain integration](https://langfuse.com/integrations/frameworks/langchain) — общий механизм callback-трейсинга.
