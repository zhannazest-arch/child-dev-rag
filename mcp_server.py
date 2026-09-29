"""
mcp_server.py — собственный MCP-сервер для RAG-пайплайна по детскому развитию.

Инструменты (tools):
  1. search_posts          — семантический поиск по базе Instagram-постов
  2. get_age_recommendations — рекомендации активностей/игрушек по возрасту ребёнка
  3. get_toy_info           — информация о конкретной игрушке или игре

Запуск сервера (stdio-транспорт, для интеграции с Claude / другими агентами):
    python mcp_server.py

Интеграция в проект:
    from mcp_server import mcp   # для тестов и прямых вызовов
"""

import re
import sys
from typing import Any

from mcp.server.fastmcp import FastMCP

# ── Ленивая инициализация компонентов RAG ─────────────────────────────────────
# Компоненты загружаются один раз при первом обращении к любому tool-у,
# чтобы не тормозить старт сервера.

_pipeline: dict | None = None


def _get_pipeline() -> dict:
    """Инициализирует и кэширует все компоненты RAG-пайплайна."""
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    # Импорты здесь — чтобы не замедлять сам импорт модуля
    from openai import OpenAI
    from main import _build_pipeline

    _pipeline = _build_pipeline(verbose=False)
    return _pipeline


# ── MCP-сервер ────────────────────────────────────────────────────────────────

mcp = FastMCP(
    name="child-dev-rag",
    instructions=(
        "MCP-сервер для поиска информации о детском развитии, игрушках и активностях. "
        "Данные основаны на Instagram-постах детских психологов и педагогов."
    ),
)


# ── Tool 1: search_posts ──────────────────────────────────────────────────────

@mcp.tool()
def search_posts(query: str, top_k: int = 5) -> list[dict[str, Any]]:
    """
    Семантический поиск по базе Instagram-постов о детском развитии.

    Использует Hybrid Retrieval (Dense + BM25 + RRF) с CrossEncoder-reranking
    и Query Rewriting для нахождения наиболее релевантных фрагментов.

    Args:
        query: Вопрос или тема для поиска (на русском языке).
               Например: "как развивать мелкую моторику у детей 2 лет"
        top_k: Количество возвращаемых фрагментов (от 1 до 10, по умолчанию 5).

    Returns:
        Список документов, каждый содержит:
        - content:  текст фрагмента
        - source:   источник (имя аккаунта Instagram)
        - score:    оценка релевантности
    """
    from reranker import retrieve_with_rewriting

    top_k = max(1, min(int(top_k), 10))
    p = _get_pipeline()

    _, docs = retrieve_with_rewriting(
        query,
        p["col_semantic"], p["bm25_index"], p["semantic_chunks"],
        p["embedding_model"], p["reranker"], p["openai_client"],
        top_k=top_k,
    )

    return [
        {
            "content": d["content"],
            "source":  d.get("metadata", {}).get("source", "unknown"),
            "score":   round(d.get("rerank_score", 0.0), 4),
        }
        for d in docs
    ]


# ── Tool 2: get_age_recommendations ──────────────────────────────────────────

@mcp.tool()
def get_age_recommendations(age_years: float) -> dict[str, Any]:
    """
    Возвращает рекомендации по активностям, игрушкам и навыкам для ребёнка
    заданного возраста на основе базы Instagram-постов.

    Args:
        age_years: Возраст ребёнка в годах (можно дробное: 1.5 = 1 год 6 месяцев).
                   Допустимый диапазон: от 0 до 7 лет.

    Returns:
        Словарь с полями:
        - age_label:       строковое обозначение возраста ("1.5 лет")
        - recommendations: список текстовых фрагментов с рекомендациями
        - answer:          сводный ответ GPT на основе найденных фрагментов
    """
    from retrieval import generate_answer
    from reranker import retrieve_with_rewriting

    age_years = max(0.0, min(float(age_years), 7.0))

    # Формируем возрастную метку для запроса
    total_months = int(round(age_years * 12))
    years  = total_months // 12
    months = total_months % 12

    if months == 0:
        age_label = f"{years} лет" if years != 1 else "1 год"
    else:
        age_label = f"{years} лет {months} месяцев" if years > 0 else f"{months} месяцев"

    query = (
        f"Какие игры, игрушки и активности подходят ребёнку {age_label}? "
        f"Какие навыки развиваются в {age_label}?"
    )

    p = _get_pipeline()
    _, docs = retrieve_with_rewriting(
        query,
        p["col_semantic"], p["bm25_index"], p["semantic_chunks"],
        p["embedding_model"], p["reranker"], p["openai_client"],
        top_k=5,
    )

    answer = generate_answer(query, docs, p["openai_client"])

    return {
        "age_label":       age_label,
        "recommendations": [d["content"][:300] for d in docs],
        "answer":          answer,
    }


# ── Tool 3: get_toy_info ──────────────────────────────────────────────────────

@mcp.tool()
def get_toy_info(toy_name: str) -> dict[str, Any]:
    """
    Ищет информацию о конкретной игрушке или игре в базе Instagram-постов.

    Возвращает: для какого возраста подходит, как использовать,
    какие навыки развивает, особенности применения.

    Args:
        toy_name: Название игрушки или игры (на русском языке).
                  Например: "пирамидка", "шнуровка", "пазлы", "рыбалка червячки"

    Returns:
        Словарь с полями:
        - toy_name:   запрошенное название
        - found:      True если найдена информация, False если нет
        - answer:     подробный ответ GPT об игрушке
        - sources:    список источников (Instagram-аккаунты)
    """
    from retrieval import generate_answer
    from reranker import retrieve_with_rewriting

    query = (
        f"Расскажи про игрушку или игру \"{toy_name}\": "
        f"для какого возраста подходит, как с ней заниматься, какие навыки развивает?"
    )

    p = _get_pipeline()
    _, docs = retrieve_with_rewriting(
        query,
        p["col_semantic"], p["bm25_index"], p["semantic_chunks"],
        p["embedding_model"], p["reranker"], p["openai_client"],
        top_k=5,
    )

    # Определяем, нашлось ли что-то реально связанное с игрушкой
    toy_lower = toy_name.lower()
    found = any(
        toy_lower in d["content"].lower()
        for d in docs
    )

    answer  = generate_answer(query, docs, p["openai_client"])
    sources = list({d.get("metadata", {}).get("source", "unknown") for d in docs})

    return {
        "toy_name": toy_name,
        "found":    found,
        "answer":   answer,
        "sources":  sources,
    }


# ── Запуск сервера ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mcp.run(transport="stdio")
