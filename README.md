# Child Development RAG Assistant

Advanced RAG-ассистент по детскому развитию на основе Instagram-постов детских психологов и педагогов.

## Возможности

- Ответы на вопросы об игрушках, играх, возрастных нормах и развитии детей от 0 до 7 лет
- Голосовой ввод (ASR через Whisper) и голосовой вывод (TTS через Minimax)
- Гибридный поиск: Dense + BM25 + RRF с CrossEncoder-reranking и Query Rewriting
- LangGraph workflow с классификацией запроса, ветвлениями и retry-циклом
- Собственный MCP-сервер с 3 инструментами для внешних AI-агентов
- Langfuse трейсинг всех LLM-вызовов

## Структура проекта

```
advancedRAG/
├── app.py                  — Gradio UI (точка входа)
├── main.py                 — инициализация пайплайна
├── pipeline_ui.py          — оркестратор запросов (ASR → Workflow → TTS)
├── workflow.py             — LangGraph граф с условной логикой
├── mcp_server.py           — MCP-сервер (3 tool-а)
├── retrieval.py            — ChromaDB, BM25, RRF, генерация ответов
├── reranker.py             — CrossEncoder, Query Rewriting
├── chunkers.py             — Naive и Semantic чанкинг
├── data_loader.py          — загрузка Instagram JSON
├── memory.py               — conversation memory
├── chat.py                 — терминальный чат
├── config.py               — настройки моделей и параметров
├── voice/tts.py            — Minimax TTS через fal.ai
├── skills/child-dev-rag/   — Agent Skill (agentskills.io)
│   └── SKILL.md
├── Evals/                  — результаты автоматической оценки
│   ├── eval.py
│   ├── eval_results.json
│   ├── eval_summary.txt
│   ├── eval_chart_bars.png
│   └── eval_chart_radar.png
├── golden_dataset.json     — 30 вопросов с эталонными ответами
├── run.ps1                 — однокомандный запуск
├── ARCHITECTURE.md         — архитектура системы
└── EVALS.md                — результаты оценки и A/B эксперимент
```

## Быстрый запуск

### 1. Настройка окружения

```bash
pip install -r requirements.txt
```

Создайте файл `.env` в папке `advancedRAG/`:

```env
OPENAI_API_KEY=sk-...
FAL_KEY=...
HF_TOKEN=...
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_BASE_URL=https://cloud.langfuse.com
```

### 2. Запуск (один скрипт)

```powershell
.\run.ps1
```

Откроет:
- **Gradio UI** на `http://127.0.0.1:7860`
- **MCP-сервер** в отдельном окне (для подключения внешних агентов)

### 3. Альтернативный запуск по отдельности

```powershell
# Только Gradio UI
$env:PYTHONIOENCODING="utf-8"; python app.py

# Только MCP-сервер
$env:PYTHONIOENCODING="utf-8"; python mcp_server.py

# Оценка качества (A/B тест, ~20 мин)
$env:PYTHONIOENCODING="utf-8"; python Evals/eval.py
```

## Модели и сервисы

| Компонент | Модель / Сервис |
|---|---|
| Embeddings | `intfloat/multilingual-e5-large` |
| Reranker | `BAAI/bge-reranker-v2-m3` |
| LLM | `gpt-4o-mini` |
| ASR | `fal-ai/whisper` |
| TTS | `fal-ai/minimax` |
| Vector DB | ChromaDB (persistent) |
| Sparse retrieval | BM25 |
| Трейсинг | Langfuse |

## Данные

RAG-база построена на Instagram-постах:
- **astrovskaya_irina** — детский психолог (143 поста)
- **kuznetsova_neuropsy** — нейропсихолог (240 постов)

Итого: 459 naive-чанков, 853 semantic-чанка.

### Сбор данных

Данные собраны в два этапа:

1. **Скрапинг** (`instagram_scraper3.py`) — Playwright-скрапер с авторизацией через сохранённую сессию. Обходит профили, собирает текст постов и URL изображений, сохраняет в JSON.

2. **OCR изображений** — посты в Instagram часто публикуются в виде слайдов (текст на картинке). Для извлечения текста из таких изображений использовался **EasyOCR** с поддержкой русского и английского языков. Распознанный текст добавляется к тексту поста перед индексацией в ChromaDB.

## MCP tools

| Tool | Описание |
|---|---|
| `search_posts(query, top_k)` | Семантический поиск по базе постов |
| `get_age_recommendations(age_years)` | Рекомендации по возрасту ребёнка |
| `get_toy_info(toy_name)` | Информация о конкретной игрушке или игре |

## Оценка качества

RAGAS-метрики на 30 вопросах golden dataset:

| Метрика | Naive RAG | Hybrid+Rerank+QR |
|---|---|---|
| Faithfulness | 0.669 | 0.592 |
| Answer Relevancy | 0.505 | 0.590 |
| Context Recall | 0.896 | 0.905 |

Подробнее — в [EVALS.md](EVALS.md).
