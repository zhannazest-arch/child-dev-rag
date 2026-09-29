"""
data_loader.py — загрузка Instagram JSON-файлов в текстовый формат для RAG.

Каждый пост превращается в текстовый блок:
    [post_id: ...]
    [date: ...]
    <текст поста>

Функция load_instagram_json() возвращает:
    source_names : List[str]  — имена файлов (без расширения)
    all_texts    : List[str]  — объединённый текст каждого аккаунта
"""

import json
from pathlib import Path
from typing import List, Tuple

from config import JSON_FILES


def load_instagram_json(
    files: List[Path] = JSON_FILES,
) -> Tuple[List[str], List[str]]:
    """
    Читает список JSON-файлов с постами Instagram.

    Returns
    -------
    source_names : список имён источников (stem файла)
    all_texts    : список текстов (один элемент = все посты одного аккаунта)
    """
    source_names: List[str] = []
    all_texts:    List[str] = []

    for path in files:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"Файл не найден: {path}\n"
                "Проверьте пути в config.py → JSON_FILES"
            )

        with open(path, encoding="utf-8") as f:
            posts = json.load(f)

        lines = []
        for post_id, post in posts.items():
            lines.append(f"[post_id: {post_id}]")
            lines.append(f"[date: {post.get('published_at', '')}]")
            lines.append(post.get("text", ""))
            lines.append("")  # разделитель между постами

        source_names.append(path.stem)
        all_texts.append("\n".join(lines))

        print(f"  {path.name}: {len(posts)} постов")

    print(f"\nИтого источников: {len(source_names)}")
    return source_names, all_texts


if __name__ == "__main__":
    names, texts = load_instagram_json()
    for name, text in zip(names, texts):
        words = len(text.split())
        print(f"  {name}: ~{words} слов")
