"""Генерирует SVG-схемы для документации.

Запуск из корня репозитория: python scripts/make_diagrams.py docs/assets
"""

import sys
from pathlib import Path

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "docs/assets")
BG_TILE, STROKE, TEXT, MUTED, LINE = "#161b22", "#2a3038", "#c9d1d9", "#8b949e", "#3d444f"
BLUE, PURPLE, GREEN, ORANGE, RED = "#a5d6ff", "#d2a8ff", "#7ee787", "#ffa657", "#ff7b72"

STYLE = (
    "<style>"
    ".s{font-family:-apple-system,'Segoe UI','Helvetica Neue','DejaVu Sans',Arial,sans-serif;font-size:13.5px;fill:%s}"
    ".h{font-weight:600}"
    ".m{font-family:ui-monospace,'SFMono-Regular','JetBrains Mono','DejaVu Sans Mono',Consolas,monospace;font-size:11.5px;fill:%s}"
    ".c{font-family:ui-monospace,'SFMono-Regular','JetBrains Mono','DejaVu Sans Mono',Consolas,monospace;font-size:11px;font-weight:700;letter-spacing:2px;fill:%s}"
    "</style>" % (TEXT, MUTED, MUTED)
)


def esc(t):
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def svg(w, h, label, body):
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" '
        f'role="img" aria-label="{esc(label)}">{STYLE}'
        f'<defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
        f'<path d="M0 0L10 5L0 10z" fill="{LINE}"/></marker></defs>{body}</svg>\n'
    )


def tile(x, y, w, h, title, sub="", accent=None):
    out = f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="{BG_TILE}" stroke="{STROKE}"/>'
    if accent:
        out += f'<rect x="{x}" y="{y + 14}" width="3" height="{h - 28}" rx="1.5" fill="{accent}"/>'
    if sub:
        out += f'<text class="s h" x="{x + 16}" y="{y + h / 2 - 3}">{esc(title)}</text>'
        out += f'<text class="m" x="{x + 16}" y="{y + h / 2 + 15}">{esc(sub)}</text>'
    else:
        out += f'<text class="s h" x="{x + 16}" y="{y + h / 2 + 5}">{esc(title)}</text>'
    return out


def arrow(x1, y1, x2, y2):
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{LINE}" stroke-width="1.5" marker-end="url(#a)"/>'


def path(d, dashed=False):
    dash = ' stroke-dasharray="4 4"' if dashed else ""
    return (
        f'<path d="{d}" fill="none" stroke="{LINE}" stroke-width="1.5"{dash} marker-end="url(#a)"/>'
    )


def cylinder(x, y, w, h, title, sub, color):
    ry = 9
    out = (
        f'<path d="M{x} {y + ry} v{h - 2 * ry} a{w / 2} {ry} 0 0 0 {w} 0 v{-(h - 2 * ry)}" '
        f'fill="{BG_TILE}" stroke="{STROKE}"/>'
    )
    out += f'<ellipse cx="{x + w / 2}" cy="{y + ry}" rx="{w / 2}" ry="{ry}" fill="{BG_TILE}" stroke="{color}"/>'
    out += f'<text class="s h" x="{x + w / 2}" y="{y + h / 2 + 6}" text-anchor="middle">{esc(title)}</text>'
    out += f'<text class="m" x="{x + w / 2}" y="{y + h / 2 + 23}" text-anchor="middle">{esc(sub)}</text>'
    return out


# --- 1. Индексация -----------------------------------------------------------
W = 820
b = '<text class="c" x="2" y="14">ИНДЕКСАЦИЯ</text>'
y, h = 34, 64
nodes = [
    (2, 128, "Документы", "pdf docx md txt", None),
    (162, 140, "Загрузка", "страницы, заголовки", None),
    (334, 140, "Нарезка", "с перекрытием", None),
    (506, 140, "Эмбеддинги", "Ollama · bge-m3", PURPLE),
]
for x, w, t, s, a in nodes:
    b += tile(x, y, w, h, t, s, a)
for (x1, w1, *_), (x2, *__) in zip(nodes, nodes[1:]):
    b += arrow(x1 + w1 + 4, y + h / 2, x2 - 6, y + h / 2)
b += arrow(650, y + h / 2, 684, y + h / 2)
b += cylinder(688, y - 6, 130, h + 12, "index.db", "SQLite", BLUE)
b += path(f"M66 {y + h + 4} V {y + h + 26} H 744 V {y + h + 10}", dashed=True)
b += f'<rect x="250" y="{y + h + 17}" width="320" height="18" fill="none"/>'
b += (
    f'<text class="m" x="410" y="{y + h + 44}" text-anchor="middle">'
    f"SHA-256 каждого файла: неизменённые пропускаются, удалённые убираются</text>"
)
(OUT / "indexing.svg").write_text(svg(W, 150, "Схема индексации документов", b), encoding="utf-8")

# --- 2. Ответ на вопрос -------------------------------------------------------
head = '<text class="c" x="2" y="14">ОТВЕТ НА ВОПРОС</text>'
b = tile(2, 92, 110, 60, "Вопрос")
b += tile(152, 46, 150, 58, "BM25", "слова и их формы", ORANGE)
b += tile(152, 140, 150, 58, "Векторы", "близость по смыслу", PURPLE)
b += tile(342, 92, 140, 60, "Слияние", "RRF, топ-k", GREEN)
b += tile(522, 92, 150, 60, "LLM", "Ollama · qwen3", PURPLE)
b += tile(712, 92, 106, 60, "Ответ", "текст [1] [2]", BLUE)
b += path("M112 112 C 132 112, 132 75, 148 75")
b += path("M112 132 C 132 132, 132 169, 148 169")
b += path("M302 75 C 322 75, 322 112, 338 112")
b += path("M302 169 C 322 169, 322 132, 338 132")
b += arrow(482, 122, 516, 122)
b += arrow(672, 122, 706, 122)
b += path("M57 92 V 30 H 597 V 86", dashed=True)
b += '<text class="m" x="330" y="25" text-anchor="middle">вопрос идёт в промпт вместе с пронумерованными фрагментами</text>'
b += '<text class="m" x="227" y="222" text-anchor="middle">оба читают index.db</text>'
b = head + f'<g transform="translate(0 24)">{b}</g>'
(OUT / "answer.svg").write_text(svg(W, 256, "Схема ответа на вопрос", b), encoding="utf-8")

# --- 3. Как работает слияние RRF ------------------------------------------
items = {
    "A": ("sql · Ранжирование", BLUE),
    "B": ("sql · Накопительные итоги", PURPLE),
    "C": ("sql · Зачем они нужны", GREEN),
    "D": ("data-quality · Пороги", ORANGE),
    "E": ("git · Коммиты", MUTED),
    "F": ("ab-testing · Анализ", MUTED),
}


def row(x, y, w, key, right=""):
    name, color = items[key]
    out = (
        f'<rect x="{x}" y="{y}" width="{w}" height="34" rx="8" fill="{BG_TILE}" stroke="{STROKE}"/>'
    )
    out += f'<circle cx="{x + 18}" cy="{y + 17}" r="9" fill="none" stroke="{color}"/>'
    out += f'<text class="m" x="{x + 18}" y="{y + 21}" text-anchor="middle" style="fill:{color}">{key}</text>'
    out += f'<text class="s" x="{x + 36}" y="{y + 22}" style="font-size:12.5px">{esc(name)}</text>'
    if right:
        out += (
            f'<text class="m" x="{x + w - 10}" y="{y + 21}" text-anchor="end">{esc(right)}</text>'
        )
    return out


b = '<text class="c" x="2" y="14">ГИБРИДНЫЙ ПОИСК: RECIPROCAL RANK FUSION</text>'
b += '<text class="m" x="2" y="38">запрос: «Чем RANK отличается от DENSE_RANK?»</text>'
cols = [(2, "BM25", ["A", "B", "D", "E"]), (262, "ВЕКТОРЫ", ["A", "C", "B", "F"])]
for x, title, keys in cols:
    b += f'<text class="c" x="{x}" y="70">{title}</text>'
    for i, k in enumerate(keys):
        b += f'<text class="m" x="{x}" y="{100 + i * 44}">{i + 1}</text>'
        b += row(x + 16, 82 + i * 44, 222, k)
fused = [
    ("A", "1/61 + 1/61 = 0.0328"),
    ("B", "1/62 + 1/63 = 0.0320"),
    ("C", "1/62 = 0.0161"),
    ("D", "1/63 = 0.0159"),
]
b += '<text class="c" x="530" y="70">ИТОГ</text>'
for i, (k, f) in enumerate(fused):
    b += f'<text class="m" x="530" y="{100 + i * 44}">{i + 1}</text>'
    b += row(546, 82 + i * 44, 272, k, "")
    b += f'<text class="m" x="{546 + 272 - 10}" y="{82 + i * 44 + 21}" text-anchor="end" style="fill:{TEXT}">{f.split("= ")[1]}</text>'
b += f'<line x1="510" y1="62" x2="510" y2="250" stroke="{STROKE}"/>'
b += (
    '<text class="m" x="2" y="282">score = Σ 1 / (60 + место в списке). '
    "B не первый ни в одном списке, но найден обоими методами и обходит C</text>"
)
(OUT / "rrf.svg").write_text(
    svg(W, 296, "Пример слияния результатов BM25 и векторного поиска", b), encoding="utf-8"
)
print("ok")
