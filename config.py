"""
config.py — централизованные настройки проекта.
Все пути, модели и константы меняются только здесь.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ── Загрузка .env ─────────────────────────────────────────────────────────────
load_dotenv(Path(__file__).parent / ".env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError(
        "OPENAI_API_KEY не найден. "
        "Добавьте OPENAI_API_KEY=sk-... в файл advancedRAG/.env"
    )

# HuggingFace token — нужен для скачивания моделей (embedding + reranker)
HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN        # для huggingface_hub
    os.environ["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN  # для старых версий transformers

# fal.ai — нужен для TTS (Minimax) и ASR (Whisper)
FAL_KEY = os.getenv("FAL_KEY")
if FAL_KEY:
    os.environ["FAL_KEY"] = FAL_KEY

# ── Пути к данным ─────────────────────────────────────────────────────────────
# Корневая папка проекта (Instagram scrapper)
PROJECT_ROOT = Path(__file__).parent.parent

# JSON-файлы с постами Instagram (добавляйте новые аккаунты сюда)
JSON_FILES = [
    PROJECT_ROOT / "astrovskaya_irina.json",
    PROJECT_ROOT / "instagram_slides_kuznetsova_neuropsy.json",
]

# Папка для сохранения новых файлов скрапера
SCRAPER_OUTPUT_DIR = PROJECT_ROOT

# Папка с авторизацией Playwright
INSTAGRAM_USER_DATA_DIR = PROJECT_ROOT / "instagram_user_data"

# ── Модели ────────────────────────────────────────────────────────────────────
EMBEDDING_MODEL = "intfloat/multilingual-e5-large"
RERANKER_MODEL  = "BAAI/bge-reranker-v2-m3"
CHAT_MODEL      = "gpt-4o-mini"

# ── Параметры чанкинга ────────────────────────────────────────────────────────
NAIVE_CHUNK_SIZE    = 1024
NAIVE_CHUNK_OVERLAP = 200

SEMANTIC_CHUNK_SIZE    = 1000
SEMANTIC_CHUNK_OVERLAP = 150

LAYOUT_MAX_PARA_TOKENS = 800
LAYOUT_OVERLAP         = 100

# ── Параметры поиска ──────────────────────────────────────────────────────────
DENSE_TOP_K   = 5
HYBRID_TOP_K  = 5
RERANK_TOP_K  = 5
RERANK_POOL   = 15
QR_VARIANTS   = 3
RRF_K         = 60
