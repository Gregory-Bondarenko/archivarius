"""Нормализация и токенизация текста для лексического поиска.

Русские слова приводятся к основе стеммером Портера (Snowball),
латинские обрабатываются английским стеммером.
"""

from __future__ import annotations

import re
from functools import lru_cache

import snowballstemmer

_TOKEN_RE = re.compile(r"[0-9a-zа-я]+(?:[-'][0-9a-zа-я]+)*", re.IGNORECASE)
_CYRILLIC_RE = re.compile(r"[а-я]")
_WHITESPACE_RE = re.compile(r"[ \t ]+")

_RU = snowballstemmer.stemmer("russian")
_EN = snowballstemmer.stemmer("english")

STOPWORDS = frozenset(
    """
    а без более бы был была были было быть в вам вас весь во вот все всего всех вы где да
    даже для до его ее если есть еще же за здесь и из или им их к как ко когда кто ли
    либо мне может мы на над не него нее нет ни них но ну о об однако он она они оно от
    очень по под при с со так также такой там те тем то того тоже той только том ту ты у
    уже чем что чтобы эта эти это этого этой этот я
    a an and are as at be but by for from has have in is it its of on or that the this to
    was were will with
    """.split()
)


def normalize(text: str) -> str:
    """Унифицирует переводы строк и пробелы."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _WHITESPACE_RE.sub(" ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


@lru_cache(maxsize=200_000)
def stem(word: str) -> str:
    if _CYRILLIC_RE.search(word):
        return _RU.stemWord(word)
    return _EN.stemWord(word)


def tokenize(text: str, *, use_stemming: bool = True) -> list[str]:
    """Разбивает текст на нормализованные термы без стоп-слов."""
    text = text.lower().replace("ё", "е")
    tokens = []
    for match in _TOKEN_RE.finditer(text):
        token = match.group(0)
        if len(token) < 2 and not token.isdigit():
            continue
        if token in STOPWORDS:
            continue
        tokens.append(stem(token) if use_stemming else token)
    return tokens
