"""
reranker.py — Cross-Encoder reranking и Query Rewriting.

Содержит:
  load_reranker()         — загрузка CrossEncoder
  rerank()                — переранжирование пула документов
  rewrite_query()         — генерация вариантов запроса через GPT
  retrieve_with_rewriting() — полный пайплайн QR + Hybrid + Rerank
"""

from collections import defaultdict
from typing import List, Tuple, Dict

from openai import OpenAI
from sentence_transformers import CrossEncoder

from config import RERANKER_MODEL, CHAT_MODEL, RERANK_TOP_K, RERANK_POOL, QR_VARIANTS, RRF_K
from retrieval import EmbeddingModel, hybrid_retrieve
import chromadb
from rank_bm25 import BM25Okapi


# ── CrossEncoder ──────────────────────────────────────────────────────────────

def load_reranker(model_name: str = RERANKER_MODEL) -> CrossEncoder:
    print(f"Загрузка reranker: {model_name}...")
    model = CrossEncoder(model_name, max_length=512)
    print("  → Reranker готов.")
    return model


def rerank(
    query: str,
    docs: List[dict],
    reranker: CrossEncoder,
    top_k: int = RERANK_TOP_K,
) -> List[dict]:
    """Переранжирует пул документов по паре (query, document)."""
    if not docs:
        return []
    pairs  = [(query, d["content"]) for d in docs]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [{**d, "rerank_score": float(sc)} for d, sc in ranked[:top_k]]


# ── Query Rewriting ───────────────────────────────────────────────────────────

def rewrite_query(
    query: str,
    client: OpenAI,
    n_variants: int = QR_VARIANTS,
    model: str = CHAT_MODEL,
) -> List[str]:
    """
    Генерирует n_variants переформулировок запроса через GPT.
    Помогает найти релевантные фрагменты, не совпадающие с исходной фразой.
    """
    prompt = (
        f"Ты — эксперт по информационному поиску.\n"
        f"Пользователь задал вопрос: \"{query}\"\n\n"
        f"Создай {n_variants} варианта переформулировки этого вопроса, "
        f"которые помогут найти более релевантную информацию. "
        f"Используй синонимы, уточнения и разные ключевые слова. "
        f"Верни ТОЛЬКО список запросов, по одному на строке, без нумерации."
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=300,
    )
    lines = [
        line.strip()
        for line in resp.choices[0].message.content.strip().split("\n")
        if line.strip()
    ]
    return lines[:n_variants]


# ── Полный пайплайн: QR + Hybrid + Rerank ────────────────────────────────────

def retrieve_with_rewriting(
    query: str,
    collection: chromadb.Collection,
    bm25: BM25Okapi,
    chunks: List[str],
    emb_model: EmbeddingModel,
    reranker: CrossEncoder,
    client: OpenAI,
    top_k: int = RERANK_TOP_K,
    pool_per_query: int = 8,
) -> Tuple[List[str], List[dict]]:
    """
    Полный пайплайн с Query Rewriting:
      1. GPT генерирует 3 переформулировки
      2. Hybrid Retrieval для каждого запроса (исходный + 3 варианта)
      3. RRF объединяет все пулы
      4. CrossEncoder переранжирует финальный пул

    Returns
    -------
    variants  : список переформулировок
    reranked  : top_k документов после rerank
    """
    variants    = rewrite_query(query, client)
    all_queries = [query] + variants

    rrf_scores:  Dict[str, float] = defaultdict(float)
    content_map: Dict[str, dict]  = {}

    for q in all_queries:
        pool = hybrid_retrieve(
            q, collection, bm25, chunks, emb_model,
            top_k=pool_per_query, dense_k=10, sparse_k=10,
        )
        for rank, doc in enumerate(pool):
            key = doc["content"][:100]
            rrf_scores[key] += 1.0 / (RRF_K + rank + 1)
            content_map[key] = doc

    fused_pool = [
        {**content_map[key], "rrf_score": sc}
        for key, sc in sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    ][:RERANK_POOL]

    reranked = rerank(query, fused_pool, reranker, top_k=top_k)
    return variants, reranked
