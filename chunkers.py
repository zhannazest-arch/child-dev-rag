"""
chunkers.py — три стратегии разбиения текста на чанки.

1. naive_chunk()    — токенный лимит + overlap (baseline)
2. semantic_chunk() — RecursiveCharacterTextSplitter (LangChain)
3. layout_chunk()   — структурный чанкинг через Docling (для PDF)

Для Instagram JSON используются naive_chunk и semantic_chunk.
layout_chunk требует объект Docling-документа и нужен при работе с PDF.
"""

from typing import List, Tuple, Any

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import (
    NAIVE_CHUNK_SIZE, NAIVE_CHUNK_OVERLAP,
    SEMANTIC_CHUNK_SIZE, SEMANTIC_CHUNK_OVERLAP,
    LAYOUT_MAX_PARA_TOKENS, LAYOUT_OVERLAP,
)


# ── Вспомогательные функции ───────────────────────────────────────────────────

def count_tokens(text: str, model: str = "gpt-4") -> int:
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))


def get_last_tokens(text: str, num_tokens: int, model: str = "gpt-4") -> str:
    enc = tiktoken.encoding_for_model(model)
    tokens = enc.encode(text)
    return enc.decode(tokens[-num_tokens:])


# ── 1. Naive Chunking ─────────────────────────────────────────────────────────

def naive_chunk(
    text: str,
    chunk_size: int = NAIVE_CHUNK_SIZE,
    overlap: int = NAIVE_CHUNK_OVERLAP,
) -> List[str]:
    """Разбивает текст по токенному лимиту с перекрытием."""
    chunks, current = [], ""
    for para in text.split("\n\n"):
        para = para.strip()
        if not para:
            continue
        candidate = (current + "\n\n" + para) if current else para
        if count_tokens(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(current)
            ov = get_last_tokens(current, overlap) if current else ""
            current = (ov + "\n\n" + para) if ov else para
    if current:
        chunks.append(current)
    return chunks


def build_naive_chunks(
    source_names: List[str],
    all_texts: List[str],
) -> Tuple[List[str], List[dict]]:
    """Применяет naive_chunk ко всем источникам."""
    chunks, metas = [], []
    for src, text in zip(source_names, all_texts):
        for ch in naive_chunk(text):
            chunks.append(ch)
            metas.append({"source": src, "method": "naive"})
    return chunks, metas


# ── 2. Semantic / Recursive Chunking ─────────────────────────────────────────

def build_semantic_chunks(
    source_names: List[str],
    all_texts: List[str],
    chunk_size: int = SEMANTIC_CHUNK_SIZE,
    chunk_overlap: int = SEMANTIC_CHUNK_OVERLAP,
) -> Tuple[List[str], List[dict]]:
    """
    Разбивает текст по иерархии разделителей:
    заголовки → абзацы → строки → предложения → слова.
    """
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " "],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    chunks, metas = [], []
    for src, text in zip(source_names, all_texts):
        for ch in splitter.split_text(text):
            chunks.append(ch)
            metas.append({"source": src, "method": "semantic"})
    return chunks, metas


# ── 3. Layout-Aware Chunking (для Docling PDF) ────────────────────────────────

def layout_chunk(
    doc: Any,
    source: str,
    max_para_tokens: int = LAYOUT_MAX_PARA_TOKENS,
    overlap: int = LAYOUT_OVERLAP,
) -> Tuple[List[str], List[dict]]:
    """
    Чанкинг на основе структуры Docling-документа.
    Таблицы никогда не разрываются; section_title фиксируется в metadata.

    Параметры
    ----------
    doc    : объект DocumentConverter().convert(...).document
    source : имя источника (stem имени файла)
    """
    chunks, metas = [], []
    current_text    = ""
    current_section = "(начало)"

    def flush(text: str, section: str, block_type: str = "paragraph") -> None:
        text = text.strip()
        if text:
            chunks.append(text)
            metas.append({
                "source":        source,
                "section_title": section,
                "block_type":    block_type,
                "method":        "layout",
            })

    for item, _ in doc.iterate_items():
        item_type = type(item).__name__

        if item_type == "SectionHeaderItem":
            flush(current_text, current_section)
            current_text    = ""
            current_section = item.text.strip()

        elif item_type == "TableItem":
            flush(current_text, current_section)
            current_text = ""
            table_md = item.export_to_markdown()
            flush(
                f"[Таблица — раздел: {current_section}]\n\n" + table_md,
                current_section,
                block_type="table",
            )

        elif item_type in ("TextItem", "ParagraphItem"):
            text = item.text.strip()
            if not text:
                continue
            candidate = (current_text + "\n\n" + text) if current_text else text
            if count_tokens(candidate) <= max_para_tokens:
                current_text = candidate
            else:
                flush(current_text, current_section)
                ov = get_last_tokens(current_text, overlap) if current_text else ""
                current_text = (ov + "\n\n" + text) if ov else text

    flush(current_text, current_section)
    return chunks, metas


def build_layout_chunks(
    docling_docs: dict,
) -> Tuple[List[str], List[dict]]:
    """Применяет layout_chunk ко всем Docling-документам."""
    chunks, metas = [], []
    for src_name, doc in docling_docs.items():
        chs, mts = layout_chunk(doc, source=src_name)
        chunks.extend(chs)
        metas.extend(mts)
    return chunks, metas
