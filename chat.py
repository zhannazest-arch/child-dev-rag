"""
chat.py — интерактивный чат с памятью диалога.

Экспортирует:
  run_chat(p) — запускает диалог в терминале.

Команды в чате:
  'выход' / 'exit' / 'quit'   — завершить сессию
  'память' / 'memory'         — показать текущую историю и резюме
  'сброс' / 'reset'           — очистить историю

Логика ответа:
  1. Если вопрос относится к истории диалога (что спрашивали, о чём говорили)
     → отвечает напрямую из памяти, без обращения к документам.
  2. Иначе → поиск по базе (Hybrid + Rerank + QR) + GPT с историей в промпте.
"""

from openai import OpenAI

from config import CHAT_MODEL
from memory import ChatMemory, generate_answer_with_memory
from reranker import retrieve_with_rewriting


# Ключевые слова, по которым определяем вопрос об истории диалога
_HISTORY_KEYWORDS = (
    "о чём", "про что", "что спрашивала", "что спрашивал",
    "о чем", "что я спрашивала", "что я спрашивал",
    "раньше спрашивала", "раньше спрашивал",
    "прошлый вопрос", "предыдущий вопрос",
    "что говорили", "о чём мы говорили", "о чем мы говорили",
    "наш разговор", "наша беседа", "история",
)


def _is_history_question(query: str) -> bool:
    """Возвращает True, если вопрос касается истории диалога, а не базы знаний."""
    q = query.lower()
    return any(kw in q for kw in _HISTORY_KEYWORDS)


def _answer_from_history(query: str, chat_memory: ChatMemory, client: OpenAI) -> str:
    """
    Отвечает на вопрос об истории разговора напрямую из памяти,
    не обращаясь к документам из базы.
    """
    history = chat_memory.get_history_text()
    if not history:
        return "У нас ещё нет истории диалога — это ваш первый вопрос."

    messages = [
        {
            "role": "system",
            "content": (
                "Ты — ассистент. Пользователь спрашивает о том, что обсуждалось ранее. "
                "Ответь кратко и точно, опираясь ТОЛЬКО на историю диалога ниже. "
                "Не придумывай ничего лишнего.\n\n"
                f"История диалога:\n{history}"
            ),
        },
        {"role": "user", "content": query},
    ]

    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        temperature=0.0,
        max_tokens=400,
    )
    answer = resp.choices[0].message.content
    # Сохраняем в память, чтобы и этот обмен был виден в следующих вопросах
    chat_memory.add(query, answer)
    return answer


def _best_retrieve(query: str, p: dict):
    """Hybrid + Rerank + QR — лучшая конфигурация поиска."""
    return retrieve_with_rewriting(
        query,
        p["col_semantic"], p["bm25_index"], p["semantic_chunks"],
        p["embedding_model"], p["reranker"], p["openai_client"],
        top_k=5,
    )[1]


def run_chat(p: dict) -> None:
    """
    Запускает интерактивный чат в терминале.

    p — словарь компонентов пайплайна (из _build_pipeline в main.py).
    """
    chat_memory = ChatMemory(max_token_limit=1000)
    client      = p["openai_client"]

    print("\n" + "=" * 65)
    print("ИНТЕРАКТИВНЫЙ ЧАТ (с памятью диалога)")
    print("Команды: 'выход' — завершить, 'память' — показать историю, 'сброс' — очистить")
    print("=" * 65)

    while True:
        try:
            query = input("\nВы: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nЧат завершён.")
            break

        if not query:
            continue

        if query.lower() in ("выход", "exit", "quit"):
            print("Чат завершён.")
            break

        if query.lower() in ("память", "memory"):
            history = chat_memory.get_history_text()
            summary = chat_memory.get_summary()
            print("\n--- Текущая история ---")
            print(history or "(пусто)")
            if summary:
                print(f"\n--- Резюме старых сообщений ---\n{summary}")
            continue

        if query.lower() in ("сброс", "reset"):
            chat_memory.clear()
            print("История очищена.")
            continue

        # Вопрос об истории диалога — отвечаем из памяти
        if _is_history_question(query):
            answer = _answer_from_history(query, chat_memory, client)
        else:
            # Обычный вопрос — поиск по базе + GPT с историей
            docs   = _best_retrieve(query, p)
            answer = generate_answer_with_memory(query, docs, client, chat_memory)

        print(f"\nАссистент: {answer}")
