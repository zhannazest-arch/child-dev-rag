"""
app.py — Gradio веб-интерфейс для Advanced RAG чата.

Запуск:
    python app.py

Откроется браузер по адресу http://localhost:7860

Пайплайн одного запроса:
    1. Ввод: текст или аудио (микрофон / файл)
    2. (если аудио) ASR → текст через Whisper (fal.ai)
    3. RAG: поиск по базе Instagram-постов + GPT с памятью диалога
    4. TTS: ответ озвучивается клонированным голосом (Minimax / fal.ai)
    5. Вывод: транскрипция вопроса + текст ответа + аудио
"""

import gradio as gr

from main import _build_pipeline
from pipeline_ui import UIPipeline

# ── Инициализация (один раз при запуске) ─────────────────────────────────────
print("Initializing RAG pipeline...")
_p        = _build_pipeline(verbose=False)
_pipeline = UIPipeline(_p)
print("Ready. Starting Gradio UI...")


# ── Обработчики событий ───────────────────────────────────────────────────────

def handle_query(audio_input, text_input):
    """Основной обработчик: принимает аудио или текст, возвращает ответ."""
    question, answer, audio_path = _pipeline.run(
        audio_path=audio_input,
        text_input=text_input,
    )
    return question, answer, audio_path


def handle_clear():
    """Сбрасывает историю диалога и очищает поля."""
    _pipeline.clear_memory()
    return "", "", "", None


def handle_show_history():
    """Возвращает текущую историю диалога для отображения."""
    return _pipeline.get_history()


# ── Интерфейс ────────────────────────────────────────────────────────────────

def build_ui():
    with gr.Blocks(title="RAG Ассистент", theme=gr.themes.Soft()) as demo:

        gr.Markdown(
            "# RAG Ассистент по детскому развитию\n"
            "Задайте вопрос текстом или голосом. "
            "Ответ будет озвучен клонированным голосом."
        )

        with gr.Row():
            # ── Левая колонка: ввод ──────────────────────────────────────────
            with gr.Column(scale=1):
                gr.Markdown("### Ввод")

                audio_input = gr.Audio(
                    label="Запишите вопрос (микрофон)",
                    sources=["microphone", "upload"],
                    type="filepath",
                )

                text_input = gr.Textbox(
                    label="Или введите текст",
                    placeholder="Например: Какие игры подходят ребёнку в 2 года?",
                    lines=3,
                )

                with gr.Row():
                    submit_btn = gr.Button("Отправить", variant="primary", size="lg")
                    clear_btn  = gr.Button("Сбросить историю", size="lg")

            # ── Правая колонка: вывод ─────────────────────────────────────────
            with gr.Column(scale=1):
                gr.Markdown("### Ответ")

                transcript_out = gr.Textbox(
                    label="Ваш вопрос (распознанный)",
                    interactive=False,
                )

                answer_out = gr.Textbox(
                    label="Ответ ассистента",
                    interactive=False,
                    lines=6,
                )

                audio_out = gr.Audio(
                    label="Озвучка ответа",
                    type="filepath",
                    autoplay=True,
                )

        # ── История диалога ───────────────────────────────────────────────────
        with gr.Accordion("История диалога", open=False):
            history_out  = gr.Textbox(
                label="",
                interactive=False,
                lines=10,
            )
            history_btn = gr.Button("Обновить историю")

        gr.Markdown(
            "---\n"
            "**Архитектура:** Embedding `intfloat/multilingual-e5-large` · "
            "Reranker `BAAI/bge-reranker-v2-m3` · "
            "LLM `gpt-4o-mini` · "
            "TTS `Minimax Speech-02-HD` (fal.ai)"
        )

        # ── Привязка событий ──────────────────────────────────────────────────
        submit_btn.click(
            fn=handle_query,
            inputs=[audio_input, text_input],
            outputs=[transcript_out, answer_out, audio_out],
        )

        # Также отправка по Enter в текстовом поле
        text_input.submit(
            fn=handle_query,
            inputs=[audio_input, text_input],
            outputs=[transcript_out, answer_out, audio_out],
        )

        clear_btn.click(
            fn=handle_clear,
            inputs=[],
            outputs=[transcript_out, answer_out, text_input, audio_out],
        )

        history_btn.click(
            fn=handle_show_history,
            inputs=[],
            outputs=[history_out],
        )

    return demo


if __name__ == "__main__":
    app = build_ui()
    app.launch(server_name="0.0.0.0", server_port=7860, inbrowser=True)
