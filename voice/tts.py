"""
voice/tts.py — синтез речи через Minimax Speech-02-HD (fal.ai).

Использует готовый voice_id из voice/voice_id.txt.
Если файл не найден — озвучка пропускается (возвращает None).

Экспортирует:
    get_voice_id()          — читает сохранённый voice_id
    generate_tts(text)      — синтезирует речь, сохраняет mp3, возвращает путь
"""

import os
import urllib.request
from pathlib import Path

import fal_client

_VOICE_ID_PATH = Path(__file__).parent / "voice_id.txt"
_TTS_MODEL     = "fal-ai/minimax/speech-02-hd"


def get_voice_id() -> str | None:
    """Читает voice_id из файла. Возвращает None если файл не найден."""
    if _VOICE_ID_PATH.exists():
        voice_id = _VOICE_ID_PATH.read_text().strip()
        if voice_id:
            return voice_id
    print("[TTS] voice_id.txt не найден — озвучка недоступна.")
    return None


def generate_tts(text: str, output_path: str = "answer_audio.mp3") -> str | None:
    """
    Синтезирует речь из текста с клонированным голосом.

    Параметры
    ----------
    text        : текст для озвучки
    output_path : путь для сохранения mp3

    Возвращает путь к файлу или None при ошибке.
    """
    voice_id = get_voice_id()
    if not voice_id:
        return None

    try:
        print(f"[TTS] Генерация озвучки: {text[:60]}...")
        result = fal_client.subscribe(
            _TTS_MODEL,
            arguments={
                "text": text,
                "voice_setting": {
                    "voice_id": voice_id,
                    "speed": 1,
                    "vol": 1,
                    "pitch": 0,
                },
                "output_format": "url",
            },
        )
        audio_url = result["audio"]["url"]
        urllib.request.urlretrieve(audio_url, output_path)
        print(f"[TTS] Аудио сохранено: {output_path}")
        return output_path
    except Exception as e:
        print(f"[TTS] Ошибка генерации: {e}")
        return None
