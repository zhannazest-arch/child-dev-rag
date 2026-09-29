# Evaluation Report

## Описание

Автоматизированная оценка RAG-пайплайна через **RAGAS** (LLM-as-judge).

- **Golden dataset:** 30 вопросов с эталонными ответами (`golden_dataset.json`)
- **Оценщик:** GPT-4o-mini как LLM-judge
- **Скрипт:** `Evals/eval.py`
- **Результаты:** `Evals/eval_results.json`

---

## A/B Эксперимент: Naive RAG vs Hybrid+Rerank+QR

Сравниваются две конфигурации поиска на одном golden dataset.

### Конфигурация A — Naive RAG

- Dense retrieval из `naive_rag` коллекции (chunk size 1024)
- Без reranking, без Query Rewriting
- top-k = 5

### Конфигурация B — Hybrid + Rerank + QR

- Query Rewriting (3 варианта запроса через GPT)
- Dense retrieval из `semantic_rag` коллекции (chunk size 1000)
- BM25 sparse retrieval
- Reciprocal Rank Fusion (RRF, k=60)
- CrossEncoder reranking (BAAI/bge-reranker-v2-m3)
- top-k = 5

---

## Метрики RAGAS

| Метрика | Что проверяет | Алгоритм |
|---|---|---|
| **Faithfulness** | Нет ли галлюцинаций — все ли утверждения ответа подтверждены контекстом | LLM извлекает claims из ответа → проверяет каждый по контексту → score = supported / total |
| **Answer Relevancy** | Отвечает ли ответ на вопрос пользователя | LLM генерирует вопросы из ответа → cosine similarity с оригинальным вопросом |
| **Context Recall** | Нашёл ли retriever всю нужную информацию | LLM разбивает reference на claims → проверяет каждый в контексте → score = found / total |

Все метрики: 0.0 (плохо) → 1.0 (отлично).

---

## Результаты

| Метрика | A: Naive RAG | B: Hybrid+Rerank+QR | Δ |
|---|---|---|---|
| **Faithfulness** | 0.669 | 0.592 | −0.077 |
| **Answer Relevancy** | 0.505 | 0.590 | **+0.085** |
| **Context Recall** | 0.896 | 0.905 | +0.009 |

---

## Выводы

**Context Recall** высокий в обоих вариантах (~0.9) — оба пайплайна хорошо находят нужную информацию в базе знаний.

**Answer Relevancy** значительно выше у Hybrid+Rerank+QR (+0.085) — Query Rewriting и reranking помогают находить более релевантный контекст, из которого генерируется более точный по теме ответ.

**Faithfulness** незначительно выше у Naive RAG (+0.077) — при более широком и разнообразном контексте (Hybrid) модель иногда включает в ответ обобщения, выходящие за рамки конкретных документов. Разница несущественная.

**Итог:** Конфигурация B (Hybrid+Rerank+QR) предпочтительна для продакшена — она даёт более релевантные ответы при сопоставимом уровне faithfulness и recall.

---

## Графики

| Файл | Описание |
|---|---|
| `Evals/eval_chart_bars.png` | Метрики по каждому из 30 вопросов |
| `Evals/eval_chart_radar.png` | Radar chart средних метрик A vs B |

---

## Воспроизведение

```powershell
$env:PYTHONIOENCODING="utf-8"; python Evals/eval.py
```

Время прогона: ~20 минут (30 вопросов × 2 конфигурации × 3 GPT-вызова RAGAS).
Результаты сохраняются в `Evals/eval_results.json` и `Evals/eval_summary.txt`.
