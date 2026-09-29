"""
memory.py — управление историей диалога.

Реализует аналог ConversationSummaryBufferMemory без устаревших пакетов:
  - Последние max_token_limit токенов хранятся дословно (буфер).
  - Более старые сообщения автоматически сжимаются в краткое резюме через GPT.
  - Это позволяет модели "помнить" длинный разговор, не превышая контекстное окно.

Экспортирует:
  ChatMemory                      — класс памяти диалога
  generate_answer_with_memory()   — аналог retrieval.generate_answer, но с историей
"""

from typing import List

from openai import OpenAI

from config import OPENAI_API_KEY, CHAT_MODEL


def _count_tokens(text: str) -> int:
    """Приблизительный подсчёт токенов (4 символа ≈ 1 токен)."""
    return len(text) // 4


class ChatMemory:
    """
    Аналог ConversationSummaryBufferMemory.

    Логика:
      1. Каждая пара (вопрос, ответ) добавляется в буфер.
      2. Если суммарный объём буфера превышает max_token_limit —
         старые сообщения сжимаются в текстовое резюме через GPT.
      3. В промпт передаётся: резюме + свежий буфер.

    Параметры
    ----------
    max_token_limit : int
        Сколько токенов хранить дословно (рекомендуется 1000–2000).
    """

    def __init__(self, max_token_limit: int = 1000):
        self.max_token_limit = max_token_limit
        self._client  = OpenAI(api_key=OPENAI_API_KEY)
        self._buffer: List[dict] = []   # список {"role": "user"/"assistant", "content": ...}
        self._summary: str = ""         # сжатое резюме старых сообщений

    # ── публичный API ─────────────────────────────────────────────────────────

    def add(self, question: str, answer: str) -> None:
        """Добавляет пару вопрос–ответ и при необходимости сжимает старые записи."""
        self._buffer.append({"role": "user",      "content": question})
        self._buffer.append({"role": "assistant", "content": answer})
        self._maybe_compress()

    def get_history_text(self) -> str:
        """
        Возвращает историю в виде текста для вставки в промпт.

        Формат:
            [Резюме предыдущих сообщений]
            ...текст резюме...

            Пользователь: ...
            Ассистент: ...
        """
        parts = []
        if self._summary:
            parts.append(f"[Резюме предыдущих сообщений]\n{self._summary}")
        for msg in self._buffer:
            label = "Пользователь" if msg["role"] == "user" else "Ассистент"
            parts.append(f"{label}: {msg['content']}")
        return "\n".join(parts)

    def get_summary(self) -> str:
        """Возвращает только текущее резюме старых сообщений."""
        return self._summary

    def clear(self) -> None:
        """Сбрасывает всю историю и резюме."""
        self._buffer  = []
        self._summary = ""

    # ── внутренняя логика ─────────────────────────────────────────────────────

    def _buffer_tokens(self) -> int:
        return sum(_count_tokens(m["content"]) for m in self._buffer)

    def _maybe_compress(self) -> None:
        """Сжимает половину буфера в резюме, если превышен лимит токенов."""
        if self._buffer_tokens() <= self.max_token_limit:
            return

        # Берём первую половину буфера для сжатия
        half      = len(self._buffer) // 2
        to_compress = self._buffer[:half]
        self._buffer = self._buffer[half:]

        # Форматируем в текст для промпта
        dialog_text = "\n".join(
            ("Пользователь: " if m["role"] == "user" else "Ассистент: ") + m["content"]
            for m in to_compress
        )

        existing = f"\nСуществующее резюме:\n{self._summary}\n" if self._summary else ""

        prompt = (
            f"Сожми следующий диалог в краткое резюме на русском языке (2–5 предложений). "
            f"Сохрани ключевые факты и темы.{existing}\n"
            f"Диалог для сжатия:\n{dialog_text}"
        )

        resp = self._client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=300,
        )
        self._summary = resp.choices[0].message.content.strip()


# ── generate_answer с памятью ─────────────────────────────────────────────────

def generate_answer_with_memory(
    query: str,
    docs: List[dict],
    client: OpenAI,
    chat_memory: ChatMemory,
    model: str = CHAT_MODEL,
) -> str:
    """
    Генерирует ответ GPT с учётом истории диалога.

    Отличие от generate_answer:
      - В системный промпт добавляется история разговора.
      - После получения ответа история автоматически обновляется.
    """
    context = "\n\n---\n\n".join(
        f"[Документ {i+1}]\n{d['content']}" for i, d in enumerate(docs)
    )

    history = chat_memory.get_history_text()
    history_block = f"\n\nИстория диалога:\n{history}" if history else ""

    messages = [
        {
            "role": "system",
            "content": (
                "Ты — полезный ассистент. "
                "Отвечай ТОЛЬКО на основе предоставленного контекста. "
                "Если информации нет — скажи об этом. "
                "Учитывай историю диалога при формулировке ответа. "
                "Если вопрос про историю чата, то дай короткий ответ из истории чата."
                "Отвечай на языке вопроса."
                + history_block
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
    answer = resp.choices[0].message.content

    chat_memory.add(query, answer)
    return answer
