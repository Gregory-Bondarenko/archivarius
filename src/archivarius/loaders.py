"""Чтение документов разных форматов.

Каждый загрузчик возвращает список сегментов: фрагмент текста и его
место в исходном документе (страница PDF или заголовок раздела).
Место попадает в цитату ответа, поэтому пользователь видит, где искать.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

from archivarius.text import normalize


@dataclass(frozen=True)
class Segment:
    text: str
    location: str = ""


@dataclass(frozen=True)
class LoadedDocument:
    path: Path
    title: str
    segments: list[Segment]

    @property
    def char_count(self) -> int:
        return sum(len(s.text) for s in self.segments)


class UnsupportedFormatError(ValueError):
    pass


_MD_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")


def _read_text_file(path: Path) -> str:
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "cp1251"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def load_txt(path: Path) -> list[Segment]:
    return [Segment(normalize(_read_text_file(path)))]


def load_markdown(path: Path) -> list[Segment]:
    """Делит markdown по заголовкам, сохраняя путь вида «Раздел / Подраздел»."""
    segments: list[Segment] = []
    trail: list[tuple[int, str]] = []
    buffer: list[str] = []
    in_code = False

    def flush() -> None:
        text = normalize("\n".join(buffer))
        # Раздел из одного заголовка без текста не несёт информации: заголовок
        # и так попадёт в место у вложенных разделов
        heading_only = bool(trail) and text == trail[-1][1]
        if text and not heading_only:
            segments.append(Segment(text, " / ".join(t for _, t in trail)))
        buffer.clear()

    for line in _read_text_file(path).splitlines():
        if line.strip().startswith("```"):
            in_code = not in_code
        match = None if in_code else _MD_HEADING_RE.match(line)
        if match:
            flush()
            level, title = len(match.group(1)), match.group(2).strip()
            trail = [(lvl, t) for lvl, t in trail if lvl < level]
            trail.append((level, title))
            buffer.append(title)
        else:
            buffer.append(line)
    flush()
    return segments


def load_pdf(path: Path) -> list[Segment]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    segments = []
    for number, page in enumerate(reader.pages, start=1):
        text = normalize(page.extract_text() or "")
        if text:
            segments.append(Segment(text, f"стр. {number}"))
    return segments


def load_docx(path: Path) -> list[Segment]:
    import docx

    document = docx.Document(str(path))
    segments: list[Segment] = []
    heading = ""
    buffer: list[str] = []

    def flush() -> None:
        text = normalize("\n\n".join(buffer))
        if text:
            segments.append(Segment(text, heading))
        buffer.clear()

    for paragraph in document.paragraphs:
        style = (paragraph.style.name if paragraph.style is not None else "").lower()
        text = paragraph.text.strip()
        if not text:
            continue
        if style.startswith(("heading", "заголовок", "title")):
            flush()
            heading = text
        buffer.append(text)
    flush()

    for index, table in enumerate(document.tables, start=1):
        rows = [" | ".join(cell.text.strip() for cell in row.cells) for row in table.rows]
        text = normalize("\n".join(rows))
        if text:
            segments.append(Segment(text, f"таблица {index}"))
    return segments


LOADERS: dict[str, Callable[[Path], list[Segment]]] = {
    ".txt": load_txt,
    ".md": load_markdown,
    ".markdown": load_markdown,
    ".pdf": load_pdf,
    ".docx": load_docx,
}

SUPPORTED_EXTENSIONS = frozenset(LOADERS)


def load_document(path: Path) -> LoadedDocument:
    loader = LOADERS.get(path.suffix.lower())
    if loader is None:
        raise UnsupportedFormatError(f"Формат не поддерживается: {path.suffix}")
    return LoadedDocument(path=path, title=path.name, segments=loader(path))


def iter_documents(root: Path) -> Iterator[Path]:
    """Обходит каталог рекурсивно, пропуская скрытые файлы и папки."""
    if root.is_file():
        if root.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield root
        return
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if any(part.startswith((".", "~$")) for part in relative.parts):
            continue
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield path
