"""Лексический поиск Okapi BM25 на инвертированном индексе."""

from __future__ import annotations

import math
from collections import Counter, defaultdict

import numpy as np

from archivarius.text import tokenize


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._postings: dict[str, list[tuple[int, int]]] = {}
        self._idf: dict[str, float] = {}
        self._doc_len = np.zeros(0, dtype=np.float32)
        self._avg_len = 0.0

    @property
    def size(self) -> int:
        return int(self._doc_len.shape[0])

    @property
    def vocabulary_size(self) -> int:
        return len(self._postings)

    def fit(self, texts: list[str]) -> BM25Index:
        postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        lengths = np.zeros(len(texts), dtype=np.float32)
        for doc_id, text in enumerate(texts):
            terms = tokenize(text)
            lengths[doc_id] = len(terms)
            for term, tf in Counter(terms).items():
                postings[term].append((doc_id, tf))
        n_docs = len(texts)
        self._postings = dict(postings)
        self._idf = {
            term: math.log(1.0 + (n_docs - len(p) + 0.5) / (len(p) + 0.5))
            for term, p in self._postings.items()
        }
        self._doc_len = lengths
        self._avg_len = float(lengths.mean()) if n_docs else 0.0
        return self

    def scores(self, query: str) -> np.ndarray:
        """Возвращает оценку BM25 для каждого документа корпуса."""
        result = np.zeros(self.size, dtype=np.float32)
        if not self.size or self._avg_len == 0:
            return result
        norm = self.k1 * (1 - self.b + self.b * self._doc_len / self._avg_len)
        for term in set(tokenize(query)):
            posting = self._postings.get(term)
            if not posting:
                continue
            ids = np.fromiter((d for d, _ in posting), dtype=np.int64, count=len(posting))
            tf = np.fromiter((t for _, t in posting), dtype=np.float32, count=len(posting))
            result[ids] += self._idf[term] * tf * (self.k1 + 1) / (tf + norm[ids])
        return result

    def top(self, query: str, k: int) -> list[tuple[int, float]]:
        scores = self.scores(query)
        positive = np.flatnonzero(scores > 0)
        if positive.size == 0:
            return []
        order = positive[np.argsort(-scores[positive], kind="stable")][:k]
        return [(int(i), float(scores[i])) for i in order]
