"""
pipeline_ui.py — оркестратор для Gradio UI.

Пайплайн одного запроса:
    1. (опционально) ASR: аудио → текст через Whisper (fal.ai)
    2. RAG:  поиск по базе + GPT с памятью диалога
    3. TTS:  текст → речь через Minimax (fal.ai), с кэшированным voice_id

Экспортирует:
    UIPipeline — класс, хранящий состояние (компоненты RAG + память чата)
"""

import os
import time
import fal_client
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from langfuse import observe, Langfuse

from config import OPENAI_API_KEY, CHAT_MODEL
from memory import ChatMemory
from workflow import run_workflow
from voice.tts import generate_tts

_ASR_MODEL = "fal-ai/whisper"

# Глобальный клиент Langfuse (для flush)
_langfuse = Langfuse()


class UIPipeline:
    """
    Хранит все компоненты RAG-пайплайна и память чата между запросами.

    Параметры
    ----------
    p : dict — словарь компонентов из main._build_pipeline()
    """

    def __init__(self, p: dict):
        self._p      = p
        self._memory = ChatMemory(max_token_limit=1000)
        self._client = p["openai_client"]

    # ── публичный метод ───────────────────────────────────────────────────────

    @observe(name="ui-request")
    def run(
        self,
        audio_path: str | None,
        text_input: str | None,
    ) -> tuple[str, str, str | None]:
        """
        Обрабатывает один запрос пользователя.

        Параметры
        ----------
        audio_path : путь к записанному аудио (или None)
        text_input : текстовый вопрос (или None)

        Возвращает
        ----------
        (question, answer, audio_output_path)
        audio_output_path — путь к mp3 или None если TTS недоступен
        """
        # ── 1. ASR ────────────────────────────────────────────────────────────
        question = self._transcribe(audio_path) if audio_path else (text_input or "").strip()

        if not question:
            _langfuse.flush()
            return "", "Введите вопрос или запишите аудио.", None

        print(f"\n[UI] Вопрос: {question}")
        _langfuse.update_current_span(input={"question": question})

        # ── 2. LangGraph workflow ─────────────────────────────────────────────
        answer = self._run_workflow_traced(question)

        print(f"[UI] Ответ: {answer[:80]}...")

        # ── 3. TTS ────────────────────────────────────────────────────────────
        audio_out = self._run_tts_traced(answer)

        _langfuse.update_current_span(output={"answer": answer})
        _langfuse.flush()

        return question, answer, audio_out

    @observe(name="workflow")
    def _run_workflow_traced(self, question: str) -> str:
        """LangGraph workflow с Langfuse трейсингом."""
        answer = run_workflow(question, self._p, self._memory)
        _langfuse.update_current_span(
            input={"question": question},
            output={"answer": answer},
        )
        return answer

    @observe(name="tts")
    def _run_tts_traced(self, answer: str) -> str | None:
        """TTS с Langfuse трейсингом."""
        _langfuse.update_current_span(input={"text_length": len(answer)})
        audio_out = generate_tts(answer, output_path="ui_answer_audio.mp3")
        _langfuse.update_current_span(output={"audio_path": audio_out})
        return audio_out

    def clear_memory(self) -> None:
        """Сбрасывает историю диалога."""
        self._memory.clear()

    def get_history(self) -> str:
        """Возвращает текущую историю для отображения в UI."""
        return self._memory.get_history_text() or "(история пуста)"

    # ── внутренние методы ─────────────────────────────────────────────────────

    @observe(name="asr")
    def _transcribe(self, audio_path: str) -> str:
        """ASR: аудио → текст через Whisper (fal.ai)."""
        print(f"[ASR] Распознавание: {audio_path}")
        _langfuse.update_current_span(input={"audio_path": audio_path, "model": _ASR_MODEL})
        try:
            fal_key = os.getenv("FAL_KEY")
            if not fal_key:
                print("[ASR] FAL_KEY не задан — ASR недоступен.")
                return ""
            audio_url = fal_client.upload_file(audio_path)
            result    = fal_client.subscribe(
                _ASR_MODEL,
                arguments={"audio_url": audio_url, "task": "transcribe"},
            )
            text = result.get("text", "").strip()
            print(f"[ASR] Распознано: {text}")
            _langfuse.update_current_span(output={"text": text})
            return text
        except Exception as e:
            print(f"[ASR] Ошибка: {e}")
            _langfuse.update_current_span(output={"error": str(e)})
            return ""
