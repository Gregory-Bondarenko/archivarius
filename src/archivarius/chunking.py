"""Нарезка сегментов на фрагменты (чанки) с перекрытием.

Границы фрагментов по возможности совпадают с абзацами, затем с
предложениями и только в крайнем случае со словами. Перекрытие
формируется из целых единиц (абзацев или предложений) с конца
предыдущего фрагмента, чтобы контекст на стыке не терялся.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from archivarius.loaders import Segment

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?…])\s+(?=[\"«(\[]?[A-ZА-ЯЁ0-9])")
_PARAGRAPH_SPLIT_RE = re.compile(r"\n\s*\n")


@dataclass(frozen=True)
class Chunk:
    text: str
    location: str
    ord: int


def split_units(text: str, max_len: int) -> list[str]:
    """Делит текст на единицы длиной не более max_len символов."""
    units: list[str] = []
    for paragraph in _PARAGRAPH_SPLIT_RE.split(text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if len(paragraph) <= max_len:
            units.append(paragraph)
            continue
        for sentence in _SENTENCE_SPLIT_RE.split(paragraph):
            sentence = sentence.strip()
            if not sentence:
                continue
            if len(sentence) <= max_len:
                units.append(sentence)
            else:
                units.extend(_split_by_words(sentence, max_len))
    return units


def _split_by_words(text: str, max_len: int) -> list[str]:
    parts: list[str] = []
    current = ""
    for word in text.split():
        while len(word) > max_len:
            if current:
                parts.append(current)
                current = ""
            parts.append(word[:max_len])
            word = word[max_len:]
        candidate = f"{current} {word}" if current else word
        if len(candidate) > max_len:
            parts.append(current)
            current = word
        else:
            current = candidate
    if current:
        parts.append(current)
    return parts


def _joined_len(units: list[str]) -> int:
    return sum(len(u) for u in units) + max(len(units) - 1, 0)


def pack_units(units: list[str], size: int, overlap: int) -> list[str]:
    """Жадно собирает единицы во фрагменты длиной около size символов."""
    chunks: list[str] = []
    current: list[str] = []
    for unit in units:
        if current and _joined_len(current) + 1 + len(unit) > size:
            chunks.append("\n".join(current))
            tail: list[str] = []
            for previous in reversed(current):
                if _joined_len([previous, *tail]) > overlap:
                    break
                tail.insert(0, previous)
            if len(tail) == len(current):
                tail = tail[1:]
            current = tail
        current.append(unit)
    if current:
        chunks.append("\n".join(current))
    return chunks


def chunk_segments(segments: list[Segment], size: int = 1000, overlap: int = 150) -> list[Chunk]:
    if overlap >= size:
        raise ValueError("overlap должен быть меньше size")
    result: list[Chunk] = []
    for segment in segments:
        for text in pack_units(split_units(segment.text, size), size, overlap):
            result.append(Chunk(text=text, location=segment.location, ord=len(result)))
    return result
