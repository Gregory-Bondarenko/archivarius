"""Фасад приложения: связывает хранилище, поиск и генерацию ответа."""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from archivarius.config import Settings
from archivarius.generation import (
    build_messages,
    extract_citations,
    filter_reasoning_stream,
    strip_reasoning,
    wants_no_think,
)
from archivarius.indexer import IndexReport, ProgressCallback, index_directory
from archivarius.models import ChatModel, Embedder, OllamaChatModel, OllamaEmbedder
from archivarius.ollama_client import OllamaClient
from archivarius.retrieval import Retriever, SearchHit
from archivarius.store import IndexStore


@dataclass
class Answer:
    question: str
    text: str
    sources: list[SearchHit]
    citations: list[int]
    timings: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.text,
            "citations": self.citations,
            "sources": [hit.to_dict() for hit in self.sources],
            "timings": self.timings,
        }


class Engine:
    def __init__(
        self,
        settings: Settings | None = None,
        embedder: Embedder | None = None,
        llm: ChatModel | None = None,
        store: IndexStore | None = None,
    ) -> None:
        self.settings = settings or Settings()
        self._client: OllamaClient | None = None
        if embedder is None or llm is None:
            self._client = OllamaClient(self.settings.ollama_url, self.settings.request_timeout)
        self.embedder: Embedder = embedder or OllamaEmbedder(
            self._client, self.settings.embed_model, self.settings.embed_batch_size
        )
        self.llm: ChatModel = llm or OllamaChatModel(
            self._client,
            self.settings.llm_model,
            temperature=self.settings.temperature,
            num_ctx=self.settings.num_ctx,
            think=self.settings.think,
        )
        self.store = store or IndexStore(self.settings.index_path)
        self.retriever = Retriever(self.store, self.embedder, self.settings)

    @classmethod
    def from_config(cls, path: str | Path | None = None) -> Engine:
        return cls(Settings.load(path))

    def close(self) -> None:
        self.store.close()
        if self._client is not None:
            self._client.close()

    def __enter__(self) -> Engine:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def index(
        self,
        path: str | Path,
        rebuild: bool = False,
        progress: ProgressCallback | None = None,
    ) -> IndexReport:
        if rebuild:
            self.store.clear()
        report = index_directory(path, self.store, self.embedder, self.settings, progress)
        self.retriever.refresh(force=True)
        return report

    def search(self, query: str, k: int | None = None, mode: str | None = None) -> list[SearchHit]:
        return self.retriever.search(query, k, mode)

    def _messages(self, question: str, hits: list[SearchHit]) -> list[dict[str, str]]:
        no_think = wants_no_think(self.llm.name, self.settings.think)
        return build_messages(question, hits, no_think=no_think)

    def ask(self, question: str, k: int | None = None, mode: str | None = None) -> Answer:
        started = time.perf_counter()
        hits = self.search(question, k, mode)
        retrieved = time.perf_counter()
        text = strip_reasoning(self.llm.complete(self._messages(question, hits)))
        finished = time.perf_counter()
        return Answer(
            question=question,
            text=text,
            sources=hits,
            citations=extract_citations(text, len(hits)),
            timings={
                "retrieval_ms": round((retrieved - started) * 1000, 1),
                "generation_ms": round((finished - retrieved) * 1000, 1),
            },
        )

    def ask_stream(
        self, question: str, k: int | None = None, mode: str | None = None
    ) -> Iterator[dict[str, Any]]:
        """Поток событий: sources, затем token-ы, в конце done с цитатами."""
        started = time.perf_counter()
        hits = self.search(question, k, mode)
        retrieved = time.perf_counter()
        yield {"type": "sources", "sources": [hit.to_dict() for hit in hits]}
        parts: list[str] = []
        hold = None if getattr(self.llm, "reasoning", False) else 300
        tokens = self.llm.stream(self._messages(question, hits))
        for token in filter_reasoning_stream(tokens, hold=hold):
            parts.append(token)
            yield {"type": "token", "text": token}
        finished = time.perf_counter()
        text = "".join(parts).strip()
        yield {
            "type": "done",
            "citations": extract_citations(text, len(hits)),
            "timings": {
                "retrieval_ms": round((retrieved - started) * 1000, 1),
                "generation_ms": round((finished - retrieved) * 1000, 1),
            },
        }

    def stats(self) -> dict[str, object]:
        data = self.store.stats()
        data.update({"llm_model": self.llm.name, "search_mode": self.settings.search_mode})
        return data
