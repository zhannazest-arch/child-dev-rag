"""
scripts/test_tools.py — быстрая проверка MCP tool-ов из child-dev-rag Skill.

Запуск из корня проекта:
    python skills/child-dev-rag/scripts/test_tools.py
"""
import sys
from pathlib import Path

# Добавляем корень проекта в путь
sys.path.insert(0, str(Path(__file__).parents[3]))

from mcp_server import search_posts, get_age_recommendations, get_toy_info

def test_all():
    print("=" * 60)
    print("child-dev-rag Skill — тест MCP tools")
    print("=" * 60)

    print("\n[1] search_posts('мелкая моторика')")
    results = search_posts("мелкая моторика", top_k=2)
    for r in results:
        print(f"  [{r['score']:.3f}] {r['content'][:80]}...")

    print("\n[2] get_age_recommendations(2.0)")
    rec = get_age_recommendations(2.0)
    print(f"  Возраст: {rec['age_label']}")
    print(f"  Ответ:   {rec['answer'][:120]}...")

    print("\n[3] get_toy_info('пирамидка')")
    toy = get_toy_info("пирамидка")
    print(f"  Найдено: {toy['found']}")
    print(f"  Ответ:   {toy['answer'][:120]}...")

    print("\nВсе tool-ы работают.")

if __name__ == "__main__":
    test_all()
