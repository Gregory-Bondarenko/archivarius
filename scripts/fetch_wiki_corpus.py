"""Скачивает корпус статей русской Википедии для оценки поиска.

Список статей лежит в examples/wiki/articles.tsv, вопросы к ним в examples/wiki/eval.jsonl.
Тексты статей распространяются по лицензии CC BY-SA 4.0, поэтому в репозиторий они
не входят: скрипт сохраняет их в data/, а ссылки на источники пишет в sources.csv.

Запуск из корня репозитория:
    python scripts/fetch_wiki_corpus.py
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from pathlib import Path
from urllib.parse import quote

import httpx

API = "https://ru.wikipedia.org/w/api.php"
HEADERS = {
    "User-Agent": "archivarius-corpus/0.1 (https://github.com/Gregory-Bondarenko/archivarius)"
}
SKIP_SECTIONS = {"примечания", "литература", "ссылки", "см. также", "источники", "комментарии"}
HEADING_RE = re.compile(r"^(={2,6})\s*(.+?)\s*\1\s*$")
WORD_RE = re.compile(r"[а-яё]{3}", re.IGNORECASE)


def strip_formulas(text: str) -> str:
    """Убирает разметку формул {\\displaystyle ...}: для поиска по тексту она только шум."""
    out = []
    i = 0
    marker = "{\\displaystyle"
    while i < len(text):
        start = text.find(marker, i)
        if start == -1:
            out.append(text[i:])
            break
        out.append(text[i:start])
        depth = 0
        j = start
        while j < len(text):
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        i = j + 1
    return "".join(out)


def is_formula_fragment(line: str) -> bool:
    """Строка из разобранной формулы: короткая и без русских слов, например «λ» или «>»."""
    stripped = line.strip()
    return bool(stripped) and len(stripped) <= 40 and not WORD_RE.search(stripped)


def drop_formulas(lines: list[str]) -> list[str]:
    """Убирает обрывки формул и склеивает предложения, которые они разрывали."""
    out: list[str] = []
    glue = False
    for line in lines:
        if is_formula_fragment(line):
            while out and not out[-1].strip():
                out.pop()
            glue = bool(out) and not HEADING_RE.match(out[-1])
            continue
        if not line.strip():
            if not glue:
                out.append("")
            continue
        if glue and out and out[-1].strip():
            out[-1] = out[-1].rstrip() + " … " + line.strip().lstrip(",. ")
        else:
            out.append(line)
        glue = False
    return out


def to_markdown(title: str, extract: str) -> str:
    lines = [f"# {title}", ""]
    skipping = False
    for line in drop_formulas(strip_formulas(extract).splitlines()):
        match = HEADING_RE.match(line)
        if match:
            level, heading = len(match.group(1)), match.group(2)
            skipping = heading.strip().lower() in SKIP_SECTIONS
            if not skipping:
                lines += ["", "#" * level + " " + heading, ""]
            continue
        if not skipping:
            lines.append(line.rstrip())
    text = "\n".join(lines)
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip() + "\n"


def fetch_extract(client: httpx.Client, title: str) -> tuple[str, str] | None:
    params = {
        "action": "query",
        "format": "json",
        "formatversion": "2",
        "prop": "extracts",
        "explaintext": "1",
        "exsectionformat": "wiki",
        "redirects": "1",
        "titles": title,
    }
    page = client.get(API, params=params).json()["query"]["pages"][0]
    if page.get("missing") or not page.get("extract"):
        return None
    return page["title"], page["extract"]


def search_title(client: httpx.Client, query: str) -> str | None:
    params = {
        "action": "query",
        "format": "json",
        "list": "search",
        "srsearch": query,
        "srlimit": "1",
    }
    hits = client.get(API, params=params).json()["query"]["search"]
    return hits[0]["title"] if hits else None


def read_articles(path: Path) -> list[tuple[str, str]]:
    articles = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("#"):
            slug, title = line.split("\t", 1)
            articles.append((slug.strip(), title.strip()))
    return articles


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--articles", type=Path, default=Path("examples/wiki/articles.tsv"))
    parser.add_argument("--out", type=Path, default=Path("data/wiki-stats"))
    args = parser.parse_args()

    articles = read_articles(args.articles)
    args.out.mkdir(parents=True, exist_ok=True)
    expected = {f"{slug}.md" for slug, _ in articles}
    for stale in args.out.glob("*.md"):
        if stale.name not in expected:
            stale.unlink()
    sources = []
    problems = 0

    with httpx.Client(headers=HEADERS, timeout=30, follow_redirects=True) as client:
        for slug, title in articles:
            result = fetch_extract(client, title)
            note = ""
            if result is None:
                found = search_title(client, title)
                result = fetch_extract(client, found) if found else None
                note = f"найдено поиском: {found}" if result else "не найдено"
            if result is None:
                print(f"  ! {slug}: «{title}» не найдена")
                problems += 1
                continue
            real_title, extract = result
            text = to_markdown(real_title, extract)
            (args.out / f"{slug}.md").write_text(text, encoding="utf-8")
            url = "https://ru.wikipedia.org/wiki/" + quote(real_title.replace(" ", "_"))
            sources.append((f"{slug}.md", real_title, url))
            if real_title != title and not note:
                note = f"перенаправление: {real_title}"
            short = len(text) < 1500
            if note or short:
                problems += 1
            mark = "!" if (note or short) else "+"
            extra = "; ".join(filter(None, [note, "короткая статья" if short else ""]))
            print(f"  {mark} {slug}: {len(text):>6} символов" + (f"  ({extra})" if extra else ""))
            time.sleep(0.2)

    with (args.out / "sources.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["file", "title", "url", "license"])
        for file, title, url in sources:
            writer.writerow([file, title, url, "CC BY-SA 4.0"])

    print(f"\nСохранено {len(sources)} из {len(articles)} статей в {args.out}")
    if problems:
        print(
            f"Строк с пометкой «!»: {problems}. Их стоит проверить: статья могла оказаться не той"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
