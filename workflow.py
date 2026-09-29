"""
workflow.py — LangGraph-граф для оркестрации RAG-пайплайна.

Граф:
    START
      ↓
    [classify]   — GPT определяет тип вопроса
      ├── "toy"     → [tool_toy]    → [check_quality]
      ├── "age"     → [tool_age]    → [check_quality]
      ├── "general" → [tool_rag]    → [check_quality]
      └── "history" → [from_memory] → [generate] → END

    [check_quality]
      ├── "ok"    → [generate] → END
      └── "retry" → [tool_rag] → [check_quality]   (цикл, max 2 попытки)

Узлы используют tool-функции из mcp_server.py.
Финальный ответ генерируется через GPT с историей из ChatMemory.

Экспортирует:
    build_graph(p)          — собирает и компилирует граф
    run_workflow(query, p, memory) — запускает граф и возвращает ответ
"""

import re
from typing import Annotated, Literal
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from openai import OpenAI

from config import CHAT_MODEL, OPENAI_API_KEY
from memory import ChatMemory, generate_answer_with_memory
from chat import _answer_from_history
from mcp_server import search_posts, get_age_recommendations, get_toy_info


# ── State ─────────────────────────────────────────────────────────────────────

class WorkflowState(TypedDict):
    query:       str           # исходный вопрос пользователя
    query_type:  str           # результат classify: toy / age / general / history
    toy_name:    str           # название игрушки (если query_type == "toy")
    age_years:   float         # возраст (если query_type == "age")
    docs:        list          # найденные документы
    answer:      str           # финальный ответ
    retry_count: int           # счётчик повторов в check_quality


# ── Вспомогательные функции ───────────────────────────────────────────────────

def _openai_client() -> OpenAI:
    return OpenAI(api_key=OPENAI_API_KEY)


def _extract_age(text: str) -> float:
    """Извлекает возраст из текста. Возвращает 2.0 если не нашёл."""
    # "1.5 лет", "2 года", "18 месяцев", "полтора года"
    m = re.search(r"(\d+(?:[.,]\d+)?)\s*(?:лет|год|года|месяц|месяца|месяцев)", text.lower())
    if m:
        val = float(m.group(1).replace(",", "."))
        # если указаны месяцы — переводим в годы
        if "месяц" in text[m.start():m.end()]:
            val = val / 12
        return round(val, 2)
    return 2.0


def _extract_toy(text: str) -> str:
    """Извлекает название игрушки из текста."""
    # Ищем слово после "про", "об", "о" или берём последние значимые слова
    m = re.search(r"(?:про|об|о|игрушк[уа]?|игр[уа]?)\s+[«\"']?([^»\"'?!.]+)[»\"']?", text.lower())
    if m:
        return m.group(1).strip()
    # fallback — вернуть весь запрос как название
    return text.strip()


# ── Node 1: classify ──────────────────────────────────────────────────────────

def node_classify(state: WorkflowState) -> WorkflowState:
    """
    Классифицирует вопрос пользователя.
    Возвращает один из типов: toy / age / general / history.
    """
    query  = state["query"]
    client = _openai_client()

    prompt = (
        "Определи тип вопроса пользователя. Верни ТОЛЬКО одно слово из списка ниже.\n\n"
        "Правила классификации:\n"
        "- toy      — вопрос содержит КОНКРЕТНОЕ название игрушки или игры "
        "(пирамидка, шнуровка, пазлы, лего, рыбалка и т.п.). "
        "НЕ toy если вопрос общий ('какой пластилин выбрать', 'что купить ребёнку').\n"
        "- age      — вопрос содержит конкретный возраст ребёнка в годах или месяцах\n"
        "- history  — вопрос о предыдущих сообщениях диалога (что спрашивали, о чём говорили)\n"
        "- general  — всё остальное: советы, рекомендации, выбор без конкретного названия\n\n"
        "Примеры:\n"
        "  'расскажи про пирамидку' → toy\n"
        "  'какой пластилин выбрать' → general\n"
        "  'что делать с ребёнком 2 лет' → age\n"
        "  'о чём я спрашивала' → history\n"
        "  'как развивать мелкую моторику' → general\n\n"
        f"Вопрос: {query}\n\nТип:"
    )

    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=10,
    )
    raw  = resp.choices[0].message.content.strip().lower()
    qtype = raw if raw in ("toy", "age", "history", "general") else "general"

    print(f"[Workflow] classify → {qtype}")

    updates: dict = {"query_type": qtype, "retry_count": 0}

    if qtype == "toy":
        updates["toy_name"] = _extract_toy(query)
    elif qtype == "age":
        updates["age_years"] = _extract_age(query)

    return {**state, **updates}


# ── Node 2a: tool_toy ─────────────────────────────────────────────────────────

def node_tool_toy(state: WorkflowState) -> WorkflowState:
    """Вызывает MCP tool get_toy_info и сохраняет документы."""
    toy_name = state.get("toy_name") or _extract_toy(state["query"])
    print(f"[Workflow] tool_toy → '{toy_name}'")

    result = get_toy_info(toy_name)

    # Превращаем ответ tool-а в список doc-объектов для check_quality
    docs = [{"content": result["answer"], "source": s} for s in result["sources"]]
    return {**state, "docs": docs}


# ── Node 2b: tool_age ─────────────────────────────────────────────────────────

def node_tool_age(state: WorkflowState) -> WorkflowState:
    """Вызывает MCP tool get_age_recommendations."""
    age_years = state.get("age_years") or _extract_age(state["query"])
    print(f"[Workflow] tool_age → {age_years} лет")

    result = get_age_recommendations(age_years)

    docs = [{"content": r, "source": "age_tool"} for r in result["recommendations"]]
    return {**state, "docs": docs}


# ── Node 2c: tool_rag ─────────────────────────────────────────────────────────

def node_tool_rag(state: WorkflowState) -> WorkflowState:
    """Вызывает MCP tool search_posts (общий поиск)."""
    query = state["query"]
    print(f"[Workflow] tool_rag → '{query[:50]}'")

    results = search_posts(query, top_k=5)
    return {**state, "docs": results}


# ── Node 2d: from_memory ──────────────────────────────────────────────────────

def node_from_memory(state: WorkflowState) -> WorkflowState:
    """Отвечает на вопрос об истории диалога напрямую из памяти."""
    print("[Workflow] from_memory")
    # Ответ будет сформирован в node_generate с флагом history
    return {**state, "docs": []}


# ── Node 3: check_quality ─────────────────────────────────────────────────────

def node_check_quality(state: WorkflowState) -> WorkflowState:
    """
    Оценивает качество найденных документов.
    Если контекст пустой или слабый и retry_count < 2 — помечает для retry.
    """
    docs        = state.get("docs", [])
    retry_count = state.get("retry_count", 0)

    # Простая эвристика: если нет документов или все короткие — retry
    has_content = any(len(d.get("content", "")) > 50 for d in docs)

    if not has_content and retry_count < 2:
        print(f"[Workflow] check_quality → retry (попытка {retry_count + 1})")
        return {**state, "retry_count": retry_count + 1}

    print("[Workflow] check_quality → ok")
    return state


# ── Node 4: generate ──────────────────────────────────────────────────────────

def node_generate(state: WorkflowState, memory: ChatMemory) -> WorkflowState:
    """Генерирует финальный ответ через GPT с историей диалога."""
    query  = state["query"]
    docs   = state.get("docs", [])
    client = _openai_client()

    if state.get("query_type") == "history":
        answer = _answer_from_history(query, memory, client)
    else:
        answer = generate_answer_with_memory(query, docs, client, memory)

    print(f"[Workflow] generate → {answer[:60]}...")
    return {**state, "answer": answer}


# ── Условные рёбра (edges) ────────────────────────────────────────────────────

def edge_after_classify(state: WorkflowState) -> Literal["tool_toy", "tool_age", "tool_rag", "from_memory"]:
    qtype = state.get("query_type", "general")
    if qtype == "toy":
        return "tool_toy"
    if qtype == "age":
        return "tool_age"
    if qtype == "history":
        return "from_memory"
    return "tool_rag"


def edge_after_quality(state: WorkflowState) -> Literal["tool_rag", "generate"]:
    docs        = state.get("docs", [])
    retry_count = state.get("retry_count", 0)
    has_content = any(len(d.get("content", "")) > 50 for d in docs)

    if not has_content and retry_count <= 2:
        return "tool_rag"
    return "generate"


def edge_after_memory(state: WorkflowState) -> Literal["generate"]:
    return "generate"


# ── Сборка графа ──────────────────────────────────────────────────────────────

def build_graph(memory: ChatMemory):
    """
    Собирает и компилирует LangGraph-граф.

    memory передаётся через замыкание в node_generate,
    чтобы граф имел доступ к истории диалога.
    """

    def _generate(state: WorkflowState) -> WorkflowState:
        return node_generate(state, memory)

    graph = StateGraph(WorkflowState)

    # Регистрируем узлы
    graph.add_node("classify",      node_classify)
    graph.add_node("tool_toy",      node_tool_toy)
    graph.add_node("tool_age",      node_tool_age)
    graph.add_node("tool_rag",      node_tool_rag)
    graph.add_node("from_memory",   node_from_memory)
    graph.add_node("check_quality", node_check_quality)
    graph.add_node("generate",      _generate)

    # Рёбра
    graph.add_edge(START, "classify")

    graph.add_conditional_edges(
        "classify",
        edge_after_classify,
        {
            "tool_toy":    "tool_toy",
            "tool_age":    "tool_age",
            "tool_rag":    "tool_rag",
            "from_memory": "from_memory",
        },
    )

    graph.add_edge("tool_toy", "check_quality")
    graph.add_edge("tool_age", "check_quality")
    graph.add_edge("tool_rag", "check_quality")

    graph.add_conditional_edges(
        "check_quality",
        edge_after_quality,
        {
            "tool_rag": "tool_rag",
            "generate": "generate",
        },
    )

    graph.add_conditional_edges(
        "from_memory",
        edge_after_memory,
        {"generate": "generate"},
    )

    graph.add_edge("generate", END)

    return graph.compile()


# ── Публичный API ─────────────────────────────────────────────────────────────

def run_workflow(query: str, p: dict, memory: ChatMemory) -> str:
    """
    Запускает LangGraph-граф для одного вопроса.

    Параметры
    ----------
    query  : вопрос пользователя
    p      : компоненты RAG-пайплайна (из _build_pipeline)
    memory : история диалога (ChatMemory)

    Возвращает
    ----------
    str — финальный ответ
    """
    # Передаём p в mcp_server через глобальный кэш
    import mcp_server
    mcp_server._pipeline = p

    compiled = build_graph(memory)

    initial_state: WorkflowState = {
        "query":       query,
        "query_type":  "",
        "toy_name":    "",
        "age_years":   0.0,
        "docs":        [],
        "answer":      "",
        "retry_count": 0,
    }

    result = compiled.invoke(initial_state)
    return result.get("answer", "")
