"""Оценка качества поиска на размеченном наборе вопросов.

Формат набора: JSONL, по одному объекту на строку.
    {"question": "...", "relevant": ["git.md", "docs/sql.pdf"]}

Релевантность задаётся на уровне документов: элемент relevant совпадает
с документом, если путь документа заканчивается на эту строку. Это
делает разметку независимой от параметров нарезки на фрагменты.
"""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from archivarius.generation import strip_reasoning
from archivarius.models import ChatModel
from archivarius.retrieval import Retriever
from archivarius.store import IndexStore


@dataclass(frozen=True)
class EvalItem:
    question: str
    relevant: list[str]


@dataclass
class ModeResult:
    mode: str
    k: int
    hit_rate: float
    recall: float
    mrr: float
    ndcg: float
    latency_ms: float
    per_question: list[dict[str, Any]] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "k": self.k,
            "hit_rate": round(self.hit_rate, 4),
            "recall": round(self.recall, 4),
            "mrr": round(self.mrr, 4),
            "ndcg": round(self.ndcg, 4),
            "latency_ms": round(self.latency_ms, 1),
        }


def load_dataset(path: str | Path) -> list[EvalItem]:
    items = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        data = json.loads(line)
        relevant = data.get("relevant")
        if not data.get("question") or not relevant:
            raise ValueError(f"Строка {number}: нужны поля question и relevant")
        if isinstance(relevant, str):
            relevant = [relevant]
        items.append(EvalItem(data["question"], [_norm(r) for r in relevant]))
    return items


def save_dataset(items: list[EvalItem], path: str | Path) -> None:
    lines = [
        json.dumps({"question": i.question, "relevant": i.relevant}, ensure_ascii=False)
        for i in items
    ]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _norm(path: str) -> str:
    return path.replace("\\", "/").strip().lower()


def matches(doc_path: str, relevant: str) -> bool:
    doc = _norm(doc_path)
    return doc == relevant or doc.endswith("/" + relevant)


# --- метрики ------------------------------------------------------------------


def hit_at_k(ranked: list[bool]) -> float:
    return 1.0 if any(ranked) else 0.0


def recall_at_k(ranked_docs: list[str], relevant: list[str]) -> float:
    found = {r for r in relevant if any(matches(d, r) for d in ranked_docs)}
    return len(found) / len(relevant)


def reciprocal_rank(ranked: list[bool]) -> float:
    for rank, is_relevant in enumerate(ranked, start=1):
        if is_relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked: list[bool], n_relevant: int) -> float:
    dcg = sum(1.0 / math.log2(rank + 1) for rank, rel in enumerate(ranked, start=1) if rel)
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(n_relevant, len(ranked)) + 1))
    return dcg / ideal if ideal else 0.0


# --- прогон -------------------------------------------------------------------


def ranked_documents(retriever: Retriever, question: str, k: int, mode: str) -> list[str]:
    """Ранжированный список уникальных документов по результатам поиска фрагментов."""
    hits = retriever.search(question, k=k * 4, mode=mode)
    documents: list[str] = []
    for hit in hits:
        if hit.chunk.path not in documents:
            documents.append(hit.chunk.path)
        if len(documents) == k:
            break
    return documents


def evaluate(
    retriever: Retriever, items: list[EvalItem], modes: list[str], k: int = 5
) -> list[ModeResult]:
    if not items:
        raise ValueError("Набор для оценки пуст")
    retriever.refresh()
    results = []
    for mode in modes:
        rows = []
        latencies = []
        for item in items:
            started = time.perf_counter()
            docs = ranked_documents(retriever, item.question, k, mode)
            latencies.append((time.perf_counter() - started) * 1000)
            flags = [any(matches(d, r) for r in item.relevant) for d in docs]
            rows.append(
                {
                    "question": item.question,
                    "retrieved": [Path(d).name for d in docs],
                    "hit": hit_at_k(flags),
                    "recall": recall_at_k(docs, item.relevant),
                    "rr": reciprocal_rank(flags),
                    "ndcg": ndcg_at_k(flags, len(item.relevant)),
                }
            )
        n = len(rows)
        results.append(
            ModeResult(
                mode=mode,
                k=k,
                hit_rate=sum(r["hit"] for r in rows) / n,
                recall=sum(r["recall"] for r in rows) / n,
                mrr=sum(r["rr"] for r in rows) / n,
                ndcg=sum(r["ndcg"] for r in rows) / n,
                latency_ms=sum(latencies) / n,
                per_question=rows,
            )
        )
    return results


def questions_label(n: int) -> str:
    if n % 10 == 1 and n % 100 != 11:
        word = "вопрос"
    elif 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14:
        word = "вопроса"
    else:
        word = "вопросов"
    return f"{n} {word}"


def render_markdown(results: list[ModeResult], n_questions: int) -> str:
    k = results[0].k if results else 0
    lines = [
        f"# Оценка качества поиска ({questions_label(n_questions)}, k = {k})",
        "",
        f"| Режим | Hit@{k} | Recall@{k} | MRR@{k} | nDCG@{k} | Задержка, мс |",
        "|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r.mode} | {r.hit_rate:.3f} | {r.recall:.3f} | {r.mrr:.3f} "
            f"| {r.ndcg:.3f} | {r.latency_ms:.1f} |"
        )
    misses = [
        row["question"]
        for r in results
        if r.mode == results[-1].mode
        for row in r.per_question
        if not row["hit"]
    ]
    if misses:
        lines += ["", f"Вопросы без попадания в режиме {results[-1].mode}:", ""]
        lines += [f"- {q}" for q in misses]
    return "\n".join(lines) + "\n"


# --- синтетический набор ------------------------------------------------------

QUESTION_PROMPT = (
    "Ниже фрагмент документа. Сформулируй один конкретный вопрос, ответ на который "
    "содержится в этом фрагменте. Вопрос должен быть понятен без фрагмента и не должен "
    "дословно повторять его формулировки. Выведи только сам вопрос.\n\nФрагмент:\n{text}"
)


def generate_synthetic(
    store: IndexStore, llm: ChatModel, n: int = 30, seed: int = 42, min_chars: int = 200
) -> list[EvalItem]:
    """Генерирует вопросы по случайным фрагментам индекса с помощью LLM."""
    chunks, _ = store.load_chunks()
    candidates = [c for c in chunks if len(c.text) >= min_chars]
    if not candidates:
        raise ValueError("В индексе нет фрагментов достаточной длины")
    rng = random.Random(seed)
    sample = rng.sample(candidates, min(n, len(candidates)))
    items = []
    for chunk in sample:
        question = llm.complete(
            [{"role": "user", "content": QUESTION_PROMPT.format(text=chunk.text)}]
        )
        question = strip_reasoning(question).strip('"«» \n').split("\n")[0].strip()
        if len(question) >= 10:
            items.append(EvalItem(question, [_norm(Path(chunk.path).name)]))
    return items
