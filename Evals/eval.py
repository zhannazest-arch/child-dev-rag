"""
eval.py — автоматизированная оценка RAG-пайплайна через RAGAS.

Прогоняет golden_dataset.json (30 вопросов) через два варианта:
  A) Naive RAG       — dense_retrieve из naive_rag коллекции
  B) Hybrid+Rerank   — retrieve_with_rewriting из semantic_rag коллекции

Метрики (RAGAS, LLM-as-judge):
  - Faithfulness        — нет ли галлюцинаций (ответ верен контексту)
  - ResponseRelevancy   — отвечает ли ответ на вопрос
  - LLMContextRecall    — нашёл ли retriever нужную информацию

Вывод:
  eval_results.json          — сырые данные по каждому вопросу
  eval_summary.txt           — таблица средних метрик (A vs B)
  eval_chart_bars.png        — столбчатая диаграмма по вопросам
  eval_chart_radar.png       — radar chart средних метрик

Запуск:
    $env:PYTHONIOENCODING="utf-8"; python eval.py
"""

import json
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # без GUI — для сохранения в файл
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from datasets import Dataset as HFDataset
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from config import OPENAI_API_KEY, CHAT_MODEL
from main import _build_pipeline
from retrieval import dense_retrieve, generate_answer
from reranker import retrieve_with_rewriting

# ── Пути к файлам вывода ──────────────────────────────────────────────────────
_DIR            = Path(__file__).parent
GOLDEN_PATH     = _DIR / "golden_dataset.json"
RESULTS_PATH    = _DIR / "eval_results.json"
SUMMARY_PATH    = _DIR / "eval_summary.txt"
CHART_BARS_PATH = _DIR / "eval_chart_bars.png"
CHART_RADAR_PATH= _DIR / "eval_chart_radar.png"

# ── Конфигурации для A/B сравнения ───────────────────────────────────────────
CONFIGS = {
    "A: Naive RAG":        "naive",
    "B: Hybrid+Rerank+QR": "hybrid",
}

METRIC_COLS   = ["faithfulness", "answer_relevancy", "context_recall"]
RAGAS_METRICS = [faithfulness, answer_relevancy, context_recall]
METRIC_LABELS = ["Faithfulness", "Answer\nRelevancy", "Context\nRecall"]
COLORS        = ["#51cf66", "#4dabf7", "#ff922b"]


# ── Вспомогательные функции ───────────────────────────────────────────────────

def load_golden(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    print(f"Golden dataset: {len(data)} вопросов из {path.name}")
    return data


def run_naive(query: str, p: dict) -> tuple[str, list[str]]:
    """Naive RAG: dense_retrieve из naive_rag + generate_answer."""
    docs    = dense_retrieve(query, p["col_naive"], p["embedding_model"], top_k=5)
    answer  = generate_answer(query, docs, p["openai_client"])
    contexts = [d["content"] for d in docs]
    return answer, contexts


def run_hybrid(query: str, p: dict) -> tuple[str, list[str]]:
    """Hybrid+Rerank+QR: retrieve_with_rewriting из semantic_rag."""
    _, docs  = retrieve_with_rewriting(
        query,
        p["col_semantic"], p["bm25_index"], p["semantic_chunks"],
        p["embedding_model"], p["reranker"], p["openai_client"],
        top_k=5,
    )
    answer   = generate_answer(query, docs, p["openai_client"])
    contexts = [d["content"] for d in docs]
    return answer, contexts


def collect_samples(golden: list[dict], p: dict, mode: str) -> list[dict]:
    """
    Прогоняет все вопросы golden dataset через пайплайн.
    mode: "naive" | "hybrid"
    Возвращает список dict с полями question / answer / contexts / reference.
    """
    runner = run_naive if mode == "naive" else run_hybrid
    results = []
    total   = len(golden)

    for i, item in enumerate(golden, 1):
        q   = item["question"]
        ref = item["ground_truth"]
        print(f"  [{i:02d}/{total}] {q[:65]}")
        try:
            answer, contexts = runner(q, p)
        except Exception as e:
            print(f"    ОШИБКА: {e}")
            answer, contexts = "Ошибка при генерации ответа.", []

        results.append({
            "question":  q,
            "answer":    answer,
            "contexts":  contexts,
            "reference": ref,
            "source":    item.get("source", ""),
        })

    return results


def build_ragas_dataset(samples: list[dict]) -> HFDataset:
    """Формирует HuggingFace Dataset в формате RAGAS 0.1.x."""
    return HFDataset.from_list([
        {
            "question":          s["question"],
            "answer":            s["answer"],
            "contexts":          s["contexts"],
            "ground_truth":      s["reference"],
        }
        for s in samples
    ])


def run_ragas(dataset: HFDataset, llm, embeddings) -> pd.DataFrame:
    """Запускает RAGAS evaluate и возвращает DataFrame."""
    # Устанавливаем LLM и embeddings глобально для метрик
    for metric in RAGAS_METRICS:
        metric.llm = llm
        if hasattr(metric, "embeddings"):
            metric.embeddings = embeddings

    result = evaluate(dataset=dataset, metrics=RAGAS_METRICS)
    return result.to_pandas()


# ── Сохранение результатов ────────────────────────────────────────────────────

def save_results(all_results: dict, all_dfs: dict) -> None:
    """Сохраняет сырые данные в eval_results.json."""
    output = {}
    for config_name, samples in all_results.items():
        df  = all_dfs[config_name]
        rows = []
        for i, s in enumerate(samples):
            row = {
                "question":       s["question"],
                "reference":      s["reference"],
                "answer":         s["answer"],
                "source":         s["source"],
                "faithfulness":   round(float(df.iloc[i]["faithfulness"]),   4),
                "answer_relevancy": round(float(df.iloc[i]["answer_relevancy"]), 4),
                "context_recall": round(float(df.iloc[i]["context_recall"]), 4),
            }
            rows.append(row)
        output[config_name] = rows

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"Результаты сохранены: {RESULTS_PATH.name}")


def save_summary(all_dfs: dict) -> None:
    """Сохраняет текстовую таблицу средних метрик в eval_summary.txt."""
    lines = []
    lines.append("=" * 65)
    lines.append("RAGAS EVALUATION SUMMARY — Advanced RAG (детское развитие)")
    lines.append("=" * 65)
    lines.append(f"{'Метрика':<22} {'A: Naive RAG':>14} {'B: Hybrid+Rerank+QR':>20} {'Δ':>8}")
    lines.append("-" * 65)

    col_map = {
        "Faithfulness":       "faithfulness",
        "Answer Relevancy":   "answer_relevancy",
        "Context Recall":     "context_recall",
    }

    for label, col in col_map.items():
        a_val = all_dfs["A: Naive RAG"][col].mean()
        b_val = all_dfs["B: Hybrid+Rerank+QR"][col].mean()
        delta = b_val - a_val
        sign  = "+" if delta >= 0 else ""
        lines.append(f"{label:<22} {a_val:>14.3f} {b_val:>20.3f} {sign}{delta:>7.3f}")

    lines.append("=" * 65)
    lines.append("Метрики: 0.0 (плохо) → 1.0 (отлично), LLM-as-judge (GPT-4o-mini)")
    lines.append(f"Golden dataset: {GOLDEN_PATH.name}")

    text = "\n".join(lines)
    print("\n" + text)

    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(f"\nТаблица сохранена: {SUMMARY_PATH.name}")


# ── Графики ───────────────────────────────────────────────────────────────────

def save_chart_bars(all_dfs: dict) -> None:
    """Столбчатая диаграмма метрик по вопросам для обеих конфигураций."""
    config_names = list(all_dfs.keys())
    n_questions  = len(all_dfs[config_names[0]])
    x            = np.arange(n_questions)
    n_configs    = len(config_names)
    n_metrics    = len(METRIC_COLS)

    fig, axes = plt.subplots(1, n_metrics, figsize=(18, 6), sharey=True)
    fig.suptitle(
        "RAGAS метрики по вопросам: Naive RAG vs Hybrid+Rerank+QR",
        fontsize=14, fontweight="bold"
    )

    config_colors = ["#4dabf7", "#ff6b6b"]  # A=синий, B=красный

    for ax_idx, (col, label) in enumerate(zip(METRIC_COLS, METRIC_LABELS)):
        ax    = axes[ax_idx]
        width = 0.35

        for c_idx, (cname, color) in enumerate(zip(config_names, config_colors)):
            vals = all_dfs[cname][col].values
            offset = (c_idx - 0.5) * width
            bars = ax.bar(x + offset, vals, width, label=cname, color=color,
                          alpha=0.85, edgecolor="white", linewidth=0.5)
            for bar, val in zip(bars, vals):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.02,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=6
                )

        ax.set_title(label.replace("\n", " "), fontsize=11, fontweight="bold")
        ax.set_xlabel("Вопрос №", fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels([str(i + 1) for i in range(n_questions)], fontsize=7)
        ax.set_ylim(0, 1.2)
        ax.axhline(y=0.7, color="gray", linestyle="--", alpha=0.4, linewidth=0.8)
        ax.grid(axis="y", alpha=0.2)
        if ax_idx == 0:
            ax.set_ylabel("Score (0–1)", fontsize=9)
        if ax_idx == n_metrics - 1:
            ax.legend(fontsize=8, loc="upper right")

    plt.tight_layout()
    plt.savefig(CHART_BARS_PATH, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"График сохранён: {CHART_BARS_PATH.name}")


def save_chart_radar(all_dfs: dict) -> None:
    """Radar chart средних метрик для A vs B."""
    config_names  = list(all_dfs.keys())
    config_colors = ["#4dabf7", "#ff6b6b"]

    angles = np.linspace(0, 2 * np.pi, len(METRIC_COLS), endpoint=False).tolist()
    angles += angles[:1]  # замкнуть

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    ax.set_title(
        "RAGAS — Средние метрики\nNaive RAG vs Hybrid+Rerank+QR",
        fontsize=13, fontweight="bold", pad=20
    )

    for cname, color in zip(config_names, config_colors):
        avg = [all_dfs[cname][col].mean() for col in METRIC_COLS]
        avg_plot = avg + [avg[0]]
        ax.fill(angles, avg_plot, color=color, alpha=0.15)
        ax.plot(angles, avg_plot, color=color, linewidth=2,
                marker="o", markersize=7, label=cname)
        for angle, val in zip(angles[:-1], avg):
            ax.text(angle, val + 0.09, f"{val:.2f}",
                    ha="center", fontsize=10, fontweight="bold", color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(METRIC_LABELS, fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=10)

    plt.tight_layout()
    plt.savefig(CHART_RADAR_PATH, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Radar chart сохранён: {CHART_RADAR_PATH.name}")


# ── Точка входа ───────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("RAGAS EVALUATION — Advanced RAG пайплайн")
    print("=" * 65)

    # 1. Инициализация RAG-компонентов
    print("\n[1/4] Инициализация RAG-пайплайна...")
    p = _build_pipeline(verbose=False)

    # 2. RAGAS оценщик
    print("\n[2/4] Инициализация RAGAS (LLM-as-judge)...")
    evaluator_llm = LangchainLLMWrapper(
        ChatOpenAI(model=CHAT_MODEL, temperature=0, api_key=OPENAI_API_KEY)
    )
    evaluator_emb = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(model="text-embedding-3-small", api_key=OPENAI_API_KEY)
    )

    # 3. Загрузка golden dataset
    golden = load_golden(GOLDEN_PATH)

    # 4. Прогон для каждой конфигурации
    all_results: dict[str, list[dict]] = {}
    all_dfs:     dict[str, pd.DataFrame] = {}

    for config_name, mode in CONFIGS.items():
        print(f"\n[3/4] Прогон: {config_name}...")
        samples = collect_samples(golden, p, mode)
        all_results[config_name] = samples

        print(f"  Оценка RAGAS для {config_name}...")
        dataset = build_ragas_dataset(samples)
        df      = run_ragas(dataset, evaluator_llm, evaluator_emb)
        all_dfs[config_name] = df

    # 5. Сохранение
    print("\n[4/4] Сохранение результатов...")
    save_results(all_results, all_dfs)
    save_summary(all_dfs)
    save_chart_bars(all_dfs)
    save_chart_radar(all_dfs)

    print("\nGotovo! Файлы:")
    for path in [RESULTS_PATH, SUMMARY_PATH, CHART_BARS_PATH, CHART_RADAR_PATH]:
        print(f"  {path.name}")


if __name__ == "__main__":
    main()
