"""Сборка запроса к языковой модели и разбор цитат в ответе."""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator
from itertools import chain

from archivarius.retrieval import SearchHit

SYSTEM_PROMPT = (
    "Ты помощник, который отвечает на вопросы по предоставленным фрагментам документов.\n"
    "Правила:\n"
    "1. Опирайся только на сведения из фрагментов, не добавляй знаний со стороны.\n"
    "2. Если вопрос требует сравнения или объяснения, сопоставь сведения из нескольких "
    "фрагментов и сделай вывод из них.\n"
    "3. После каждого утверждения ставь ссылку на номер фрагмента в квадратных скобках, "
    "например [1] или [2][3].\n"
    "4. Если фрагменты отвечают на вопрос только частично, ответь на то, что в них есть, "
    "и прямо скажи, чего не хватает.\n"
    "5. Только если во фрагментах нет сведений по теме вопроса, ответь: "
    "«В предоставленных документах нет ответа на этот вопрос».\n"
    "6. Отвечай на языке вопроса, кратко и по существу."
)

_CITATION_RE = re.compile(r"\[(\d{1,3})\]")
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def format_source(index: int, hit: SearchHit) -> str:
    place = f", {hit.chunk.location}" if hit.chunk.location else ""
    return f"[{index}] {hit.chunk.title}{place}\n{hit.chunk.text}"


def build_messages(
    question: str, hits: list[SearchHit], no_think: bool = False
) -> list[dict[str, str]]:
    context = "\n\n".join(format_source(i, hit) for i, hit in enumerate(hits, start=1))
    if not context:
        context = "(подходящих фрагментов не найдено)"
    user = f"Фрагменты документов:\n\n{context}\n\nВопрос: {question.strip()}"
    if no_think:
        user += "\n\n/no_think"
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def wants_no_think(model: str, think: bool) -> bool:
    """Qwen3 рассуждает по умолчанию; директива /no_think отключает это на уровне промпта."""
    return not think and model.lower().startswith("qwen3")


def extract_citations(answer: str, n_sources: int) -> list[int]:
    """Возвращает номера процитированных источников в порядке появления."""
    seen: list[int] = []
    for match in _CITATION_RE.finditer(answer):
        number = int(match.group(1))
        if 1 <= number <= n_sources and number not in seen:
            seen.append(number)
    return seen


def strip_reasoning(text: str) -> str:
    """Убирает рассуждения модели, включая случай без открывающего тега."""
    text = _THINK_RE.sub("", text)
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    return text.strip()


def filter_reasoning_stream(tokens: Iterable[str], hold: int | None = 300) -> Iterator[str]:
    """Потоковый вариант strip_reasoning.

    Шаблон чата иногда сам открывает блок рассуждений, и модель присылает только
    закрывающий тег. Поэтому начало ответа придерживается, пока не станет ясно,
    есть ли в нём рассуждения: найден </think> или набралось hold символов без тегов.
    При hold=None ответ придерживается до </think> или до конца потока: так работают
    модели, которые рассуждают всегда.
    """
    iterator = iter(tokens)
    buffer = ""
    for token in iterator:
        buffer += token
        if "</think>" in buffer:
            buffer = buffer.rsplit("</think>", 1)[1].lstrip()
            break
        head = buffer.lstrip()
        if head.startswith("<think>") or "<think>".startswith(head):
            continue
        if hold is not None and len(buffer) >= hold:
            break
    else:
        tail = strip_reasoning(buffer)
        if tail:
            yield tail
        return
    yield from _lstrip_stream(_filter_paired(chain([buffer], iterator)))


def _lstrip_stream(tokens: Iterable[str]) -> Iterator[str]:
    """Убирает пробелы и переводы строк в самом начале потока."""
    started = False
    for token in tokens:
        if not started:
            token = token.lstrip()
            if not token:
                continue
            started = True
        yield token


def _filter_paired(tokens: Iterable[str]) -> Iterator[str]:
    """Скрывает содержимое пар <think>...</think>, даже если тег разорван между токенами."""
    buffer = ""
    inside = False
    for token in tokens:
        buffer += token
        while True:
            if inside:
                end = buffer.find("</think>")
                if end == -1:
                    buffer = buffer[-len("</think>") :]
                    break
                buffer = buffer[end + len("</think>") :].lstrip()
                inside = False
            else:
                start = buffer.find("<think>")
                if start == -1:
                    safe = len(buffer) - len("<think>")
                    if safe > 0:
                        yield buffer[:safe]
                        buffer = buffer[safe:]
                    break
                if start:
                    yield buffer[:start]
                buffer = buffer[start + len("<think>") :]
                inside = True
    if buffer and not inside:
        yield buffer
