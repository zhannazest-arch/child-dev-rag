"""
main.py — точка входа Advanced RAG Pipeline.

Запуск:
    python main.py            — интерактивный чат (по умолчанию)
    python main.py --batch    — только пакетный прогон TEST_QUESTIONS
    python main.py --both     — пакетный прогон, затем чат

Пайплайн:
    1. Загрузка Instagram JSON (config.py → JSON_FILES)
    2. Чанкинг (Naive + Semantic)
    3. Построение ChromaDB + BM25
    4. Загрузка моделей (Embedding, Reranker, OpenAI)
    5. Пакетный прогон TEST_QUESTIONS с памятью (ConversationSummaryBufferMemory)
    6. Интерактивный чат с памятью
"""

import sys

from openai import OpenAI
from tabulate import tabulate

from config import OPENAI_API_KEY, EMBEDDING_MODEL, RERANKER_MODEL, CHAT_MODEL
from data_loader import load_instagram_json
from chunkers import build_naive_chunks, build_semantic_chunks, count_tokens
from retrieval import (
    EmbeddingModel, build_chroma, get_chroma_client,
    dense_retrieve, build_bm25, hybrid_retrieve, generate_answer,
)
from reranker import load_reranker, rerank, retrieve_with_rewriting
from memory import ChatMemory, generate_answer_with_memory
from chat import run_chat


# ── Тестовые вопросы (пакетный режим) ────────────────────────────────────────
# Каждый вопрос видит историю ответов на предыдущие
TEST_QUESTIONS = [
    "Какие игры рекомендуются для развития навыка держать карандаш или ручку?",
    "Как научить ребёнка аккуратно закрашивать расскраски?",
    "Какие навыками должен быть обучен ребёнок в 3 года?",
    "Как научить ребёнка собирать пазлы?",
    "Какие настольные игры подходят ребёнку в 3 года?",
]


def _build_pipeline(verbose: bool = True):
    """
    Инициализирует все компоненты и возвращает их в словаре.

    verbose=True  — полный вывод с таблицей чанкинга (для --batch / --both)
    verbose=False — краткий вывод только о загрузке (для --chat)
    """
    print("=" * 65)
    print("ADVANCED RAG PIPELINE")
    print("=" * 65)
    print(f"Embedding : {EMBEDDING_MODEL}")
    print(f"Reranker  : {RERANKER_MODEL}")
    print(f"LLM       : {CHAT_MODEL}")

    total = 3 if not verbose else 4

    print(f"\n[1/{total}] Загрузка данных...")
    source_names, all_texts = load_instagram_json()

    print(f"\n[2/{total}] Чанкинг...")
    naive_chunks,    naive_meta    = build_naive_chunks(source_names, all_texts)
    semantic_chunks, semantic_meta = build_semantic_chunks(source_names, all_texts)

    if verbose:
        naive_sizes    = [count_tokens(c) for c in naive_chunks]
        semantic_sizes = [count_tokens(c) for c in semantic_chunks]
        rows = [
            ["Naive",    len(naive_chunks),    sum(naive_sizes)//len(naive_sizes),
             min(naive_sizes), max(naive_sizes)],
            ["Semantic", len(semantic_chunks), sum(semantic_sizes)//len(semantic_sizes),
             min(semantic_sizes), max(semantic_sizes)],
        ]
        print(tabulate(rows,
            headers=["Метод", "Чанков", "Среднее (tok)", "Мин", "Макс"],
            tablefmt="simple"))
    else:
        print(f"  Naive: {len(naive_chunks)} чанков | Semantic: {len(semantic_chunks)} чанков")

    print(f"\n[3/{total}] Инициализация моделей...")
    embedding_model = EmbeddingModel(EMBEDDING_MODEL)
    openai_client   = OpenAI(api_key=OPENAI_API_KEY)
    reranker        = load_reranker(RERANKER_MODEL)

    if verbose:
        step = f"[4/{total}]"
    else:
        step = f"[3/{total}]"

    print(f"\n{step} Построение индексов ChromaDB + BM25...")
    chroma_client = get_chroma_client()
    col_naive    = build_chroma("naive_rag",    naive_chunks,    naive_meta,    embedding_model, chroma_client)
    col_semantic = build_chroma("semantic_rag", semantic_chunks, semantic_meta, embedding_model, chroma_client)
    bm25_index   = build_bm25(semantic_chunks)
    print(f"BM25 построен: {len(semantic_chunks)} документов")

    return dict(
        embedding_model=embedding_model,
        openai_client=openai_client,
        reranker=reranker,
        col_naive=col_naive,
        col_semantic=col_semantic,
        semantic_chunks=semantic_chunks,
        bm25_index=bm25_index,
    )


def _best_retrieve(query, p):
    """Лучшая конфигурация поиска для пакетного режима: Semantic + Hybrid + Rerank + QR."""
    return retrieve_with_rewriting(
        query,
        p["col_semantic"], p["bm25_index"], p["semantic_chunks"],
        p["embedding_model"], p["reranker"], p["openai_client"],
        top_k=5,
    )[1]


def run_batch(p):
    """
    Пакетный режим: прогоняет TEST_QUESTIONS через все 4 конфигурации.
    Конфигурация 'Best (с памятью)' использует ConversationSummaryBufferMemory —
    каждый вопрос видит историю предыдущих ответов.
    """
    configurations = {
        "Naive RAG": lambda q: (
            dense_retrieve(q, p["col_naive"], p["embedding_model"], top_k=3),
            None,
        ),
        "Semantic + Hybrid + Rerank": lambda q: (
            rerank(
                q,
                hybrid_retrieve(q, p["col_semantic"], p["bm25_index"],
                                p["semantic_chunks"], p["embedding_model"],
                                top_k=12, dense_k=15, sparse_k=15),
                p["reranker"], top_k=3,
            ),
            None,
        ),
        "Best (с памятью)": None,   # обрабатывается отдельно
    }

    # Память для пакетного режима — живёт весь прогон TEST_QUESTIONS
    batch_memory = ChatMemory(max_token_limit=1000)

    print("\n" + "=" * 65)
    print("ПАКЕТНЫЙ ПРОГОН TEST_QUESTIONS")
    print("=" * 65)

    for q in TEST_QUESTIONS:
        print(f"\n{'='*65}")
        print(f"Вопрос: {q}")

        # Обычные конфигурации без памяти
        for cfg_name, retrieve_fn in configurations.items():
            if retrieve_fn is None:
                continue
            docs, _ = retrieve_fn(q)
            answer  = generate_answer(q, docs, p["openai_client"])
            print(f"\n  [{cfg_name}]")
            print(f"  {answer[:300]}")

        # Best-конфигурация с памятью
        docs   = _best_retrieve(q, p)
        answer = generate_answer_with_memory(q, docs, p["openai_client"], batch_memory)
        print(f"\n  [Best (с памятью)]")
        print(f"  {answer[:300]}")

        # Показываем текущее состояние памяти
        summary = batch_memory.get_summary()
        if summary:
            print(f"\n  [Резюме памяти]: {summary[:200]}")

    print("\n" + "=" * 65)
    print("Пакетный прогон завершён.")


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "chat"

    if mode in ("--batch", "batch"):
        p = _build_pipeline(verbose=True)
        run_batch(p)
    elif mode in ("--both", "both"):
        p = _build_pipeline(verbose=True)
        run_batch(p)
        run_chat(p)
    else:
        # --chat или без аргументов: краткая инициализация, сразу чат
        p = _build_pipeline(verbose=False)
        run_chat(p)


if __name__ == "__main__":
    main()
