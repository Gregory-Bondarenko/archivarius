"""Поиск фрагментов: лексический (BM25), векторный и гибридный.

Гибридный режим объединяет два ранжированных списка методом
Reciprocal Rank Fusion: score = Σ 1 / (k + rank). Метод не требует
калибровки шкал BM25 и косинусной близости между собой.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np

from archivarius.bm25 import BM25Index
from archivarius.config import SEARCH_MODES, Settings
from archivarius.models import Embedder
from archivarius.store import ChunkRecord, IndexStore


@dataclass(frozen=True)
class SearchHit:
    chunk: ChunkRecord
    score: float
    sparse_rank: int | None = None
    dense_rank: int | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "chunk_id": self.chunk.id,
            "title": self.chunk.title,
            "path": self.chunk.path,
            "location": self.chunk.location,
            "text": self.chunk.text,
            "score": round(self.score, 6),
            "sparse_rank": self.sparse_rank,
            "dense_rank": self.dense_rank,
        }


def reciprocal_rank_fusion(
    rankings: list[list[int]], k: int = 60, weights: list[float] | None = None
) -> list[tuple[int, float]]:
    weights = weights or [1.0] * len(rankings)
    scores: dict[int, float] = {}
    for ranking, weight in zip(rankings, weights):
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + weight / (k + rank)
    return sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))


class Retriever:
    def __init__(self, store: IndexStore, embedder: Embedder, settings: Settings) -> None:
        self.store = store
        self.embedder = embedder
        self.settings = settings
        self._lock = threading.Lock()
        self._revision: tuple[int, int] | None = None
        self._chunks: list[ChunkRecord] = []
        self._matrix = np.zeros((0, 0), dtype=np.float32)
        self._bm25 = BM25Index(settings.bm25_k1, settings.bm25_b)

    def refresh(self, force: bool = False) -> None:
        """Перечитывает индекс из хранилища, если он изменился."""
        with self._lock:
            if not force and self._revision == self.store.revision:
                return
            self._chunks, self._matrix = self.store.load_chunks()
            self._bm25 = BM25Index(self.settings.bm25_k1, self.settings.bm25_b).fit(
                [c.text for c in self._chunks]
            )
            self._revision = self.store.revision

    @property
    def size(self) -> int:
        return len(self._chunks)

    def _sparse(self, query: str, n: int) -> list[tuple[int, float]]:
        return self._bm25.top(query, n)

    def _dense(self, query: str, n: int) -> list[tuple[int, float]]:
        if not self.size:
            return []
        vector = self.embedder.embed([self.settings.query_prefix + query])[0]
        if vector.shape[0] != self._matrix.shape[1]:
            raise ValueError(
                "Размерность вектора запроса не совпадает с индексом. "
                "Проверьте embed_model или пересоберите индекс."
            )
        similarities = self._matrix @ vector
        n = min(n, similarities.shape[0])
        top = np.argpartition(-similarities, n - 1)[:n]
        top = top[np.argsort(-similarities[top], kind="stable")]
        return [(int(i), float(similarities[i])) for i in top]

    def search(self, query: str, k: int | None = None, mode: str | None = None) -> list[SearchHit]:
        k = k or self.settings.top_k
        mode = mode or self.settings.search_mode
        if mode not in SEARCH_MODES:
            raise ValueError(f"Неизвестный режим поиска: {mode}")
        if not query.strip():
            return []
        self.refresh()
        if not self.size:
            return []

        per_doc = self.settings.max_chunks_per_doc
        if mode == "bm25":
            sparse = self._diverse(self._sparse(query, k * 4), per_doc, k)
            return [
                SearchHit(self._chunks[i], s, sparse_rank=r)
                for r, (i, s) in enumerate(sparse, start=1)
            ]
        if mode == "dense":
            dense = self._diverse(self._dense(query, k * 4), per_doc, k)
            return [
                SearchHit(self._chunks[i], s, dense_rank=r)
                for r, (i, s) in enumerate(dense, start=1)
            ]

        # Кандидаты не ограничиваются по документам: несколько сильных фрагментов
        # одной статьи сами по себе сигнал релевантности. Разнообразие наводится
        # только в итоговой выдаче
        pool = max(self.settings.candidate_pool, k)
        sparse = self._sparse(query, pool)
        dense = self._dense(query, pool)
        sparse_rank = {i: r for r, (i, _) in enumerate(sparse, start=1)}
        dense_rank = {i: r for r, (i, _) in enumerate(dense, start=1)}
        fused = reciprocal_rank_fusion(
            [[i for i, _ in sparse], [i for i, _ in dense]], k=self.settings.rrf_k
        )
        return [
            SearchHit(
                self._chunks[i],
                score,
                sparse_rank=sparse_rank.get(i),
                dense_rank=dense_rank.get(i),
            )
            for i, score in self._diverse(fused, per_doc, k)
        ]

    def _diverse(
        self, ranked: list[tuple[int, float]], per_doc: int, limit: int
    ) -> list[tuple[int, float]]:
        """Оставляет не больше per_doc фрагментов из одного документа.

        Без этого длинная статья может занять всю выдачу, и вопрос, который
        сравнивает две темы, останется без второго источника. per_doc = 0 отключает ограничение.
        """
        if per_doc <= 0:
            return ranked[:limit]
        taken: dict[int, int] = {}
        result = []
        for index, score in ranked:
            doc = self._chunks[index].doc_id
            if taken.get(doc, 0) >= per_doc:
                continue
            taken[doc] = taken.get(doc, 0) + 1
            result.append((index, score))
            if len(result) == limit:
                break
        return result
