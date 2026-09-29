"""
retrieval.py — компоненты поиска для RAG-пайплайна.

Содержит:
  EmbeddingModel       — обёртка над SentenceTransformer (батч + прогресс-бар)
  build_chroma()       — создание/загрузка коллекции ChromaDB (персистентная)
  dense_retrieve()     — семантический поиск
  build_bm25()         — построение BM25-индекса
  bm25_retrieve()      — лексический поиск
  reciprocal_rank_fusion() — объединение результатов через RRF
  hybrid_retrieve()    — Dense + BM25 + RRF
  generate_answer()    — ответ через OpenAI GPT
"""

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import List, Dict

import chromadb
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from tqdm import tqdm

from config import (
    EMBEDDING_MODEL, CHAT_MODEL,
    DENSE_TOP_K, HYBRID_TOP_K, RRF_K,
)

# Папка для хранения ChromaDB на диске (рядом с этим файлом)
_CHROMA_DIR = Path(__file__).parent / ".chroma_cache"


# ── Embedding Model ───────────────────────────────────────────────────────────

class EmbeddingModel:
    def __init__(self, model_name: str = EMBEDDING_MODEL, batch_size: int = 32):
        self.model      = SentenceTransformer(model_name)
        self.batch_size = batch_size

    def embed(self, text: str) -> List[float]:
        return self.model.encode(text, convert_to_numpy=True).tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Кодирует тексты батчами с прогресс-баром."""
        all_embs = []
        for start in tqdm(range(0, len(texts), self.batch_size),
                          desc="  эмбеддинги", unit="батч"):
            batch = texts[start : start + self.batch_size]
            embs  = self.model.encode(batch, convert_to_numpy=True).tolist()
            all_embs.extend(embs)
        return all_embs


# ── ChromaDB (персистентный) ──────────────────────────────────────────────────

def _collection_hash(chunks: List[str]) -> str:
    """Короткий хэш содержимого коллекции для обнаружения изменений."""
    digest = hashlib.md5("".join(chunks).encode()).hexdigest()[:12]
    return digest


def get_chroma_client() -> chromadb.PersistentClient:
    """Возвращает персистентный клиент ChromaDB (создаёт папку при необходимости)."""
    _CHROMA_DIR.mkdir(exist_ok=True)
    return chromadb.PersistentClient(path=str(_CHROMA_DIR))


def build_chroma(
    name: str,
    chunks: List[str],
    metas: List[dict],
    emb_model: EmbeddingModel,
    client: chromadb.PersistentClient,
) -> chromadb.Collection:
    """
    Возвращает коллекцию ChromaDB.

    Логика кэширования:
    - Если коллекция уже существует И хэш чанков совпадает → загружаем с диска (быстро).
    - Если коллекции нет или данные изменились → пересчитываем эмбеддинги и сохраняем.
    """
    current_hash = _collection_hash(chunks)
    hash_file    = _CHROMA_DIR / f"{name}.hash"

    # Проверяем кэш
    if hash_file.exists():
        saved_hash = hash_file.read_text().strip()
        if saved_hash == current_hash:
            try:
                col = client.get_collection(name=name)
                print(f"  → '{name}': загружено из кэша ({col.count()} чанков)")
                return col
            except Exception:
                pass  # коллекция не найдена — пересоздаём

    # Пересоздаём коллекцию
    try:
        client.delete_collection(name)
    except Exception:
        pass

    print(f"Генерация эмбеддингов для {len(chunks)} чанков ({name})...")
    col  = client.create_collection(name=name, metadata={"hnsw:space": "cosine"})
    embs = emb_model.embed_batch(chunks)
    ids  = [f"{name}_{i}" for i in range(len(chunks))]
    col.add(documents=chunks, embeddings=embs, metadatas=metas, ids=ids)

    hash_file.write_text(current_hash)
    print(f"  → '{name}': {col.count()} чанков сохранено в кэш")
    return col


# ── Dense Retrieval ───────────────────────────────────────────────────────────

def dense_retrieve(
    query: str,
    collection: chromadb.Collection,
    emb_model: EmbeddingModel,
    top_k: int = DENSE_TOP_K,
) -> List[dict]:
    q_emb = emb_model.embed(query)
    res   = collection.query(
        query_embeddings=[q_emb],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    return [
        {
            "content":  res["documents"][0][i],
            "metadata": res["metadatas"][0][i],
            "score":    1 - res["distances"][0][i],
        }
        for i in range(len(res["documents"][0]))
    ]


# ── BM25 ──────────────────────────────────────────────────────────────────────

def build_bm25(chunks: List[str]) -> BM25Okapi:
    tokenized = [ch.lower().split() for ch in chunks]
    return BM25Okapi(tokenized)


def bm25_retrieve(
    query: str,
    bm25: BM25Okapi,
    chunks: List[str],
    top_k: int = 10,
) -> List[dict]:
    scores = bm25.get_scores(query.lower().split())
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
    return [
        {"content": chunks[idx], "bm25_score": sc, "index": idx}
        for idx, sc in ranked
    ]


# ── Reciprocal Rank Fusion ────────────────────────────────────────────────────

def reciprocal_rank_fusion(
    dense_docs: List[dict],
    sparse_docs: List[dict],
    k: int = RRF_K,
) -> List[dict]:
    rrf_scores:  Dict[str, float] = defaultdict(float)
    content_map: Dict[str, dict]  = {}

    for rank, doc in enumerate(dense_docs):
        key = doc["content"][:100]
        rrf_scores[key] += 1.0 / (k + rank + 1)
        content_map[key] = doc

    for rank, doc in enumerate(sparse_docs):
        key = doc["content"][:100]
        rrf_scores[key] += 1.0 / (k + rank + 1)
        if key not in content_map:
            content_map[key] = doc

    return [
        {**content_map[key], "rrf_score": sc}
        for key, sc in sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    ]


# ── Hybrid Retrieval ──────────────────────────────────────────────────────────

def hybrid_retrieve(
    query: str,
    collection: chromadb.Collection,
    bm25: BM25Okapi,
    chunks: List[str],
    emb_model: EmbeddingModel,
    top_k: int = HYBRID_TOP_K,
    dense_k: int = 10,
    sparse_k: int = 10,
) -> List[dict]:
    dense_res  = dense_retrieve(query, collection, emb_model, top_k=dense_k)
    sparse_res = bm25_retrieve(query, bm25, chunks, top_k=sparse_k)
    fused      = reciprocal_rank_fusion(dense_res, sparse_res)
    return fused[:top_k]


# ── LLM Answer Generation ─────────────────────────────────────────────────────

def generate_answer(
    query: str,
    docs: List[dict],
    client: OpenAI,
    model: str = CHAT_MODEL,
) -> str:
    context = "\n\n---\n\n".join(
        f"[Документ {i+1}]\n{d['content']}" for i, d in enumerate(docs)
    )
    messages = [
        {
            "role": "system",
            "content": (
                "Ты — полезный ассистент. Отвечай ТОЛЬКО на основе предоставленного контекста. "
                "Если информации нет — скажи об этом. Отвечай на языке вопроса."
            ),
        },
        {
            "role": "user",
            "content": f"Контекст:\n{context}\n\n---\n\nВопрос: {query}\n\nОтвет:",
        },
    ]
    resp = client.chat.completions.create(
        model=model, messages=messages, temperature=0.2, max_tokens=800
    )
    return resp.choices[0].message.content
