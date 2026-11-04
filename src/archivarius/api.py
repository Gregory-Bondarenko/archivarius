"""HTTP API и веб-интерфейс на FastAPI."""

from __future__ import annotations

import json
from collections.abc import Iterator
from importlib import resources

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

from archivarius import __version__
from archivarius.config import SEARCH_MODES
from archivarius.engine import Engine
from archivarius.ollama_client import OllamaError


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    k: int | None = Field(default=None, ge=1, le=50)
    mode: str | None = None
    stream: bool = False


def _check_mode(mode: str | None) -> None:
    if mode is not None and mode not in SEARCH_MODES:
        raise HTTPException(422, f"mode должен быть одним из {list(SEARCH_MODES)}")


def create_app(engine: Engine) -> FastAPI:
    app = FastAPI(title="Archivarius", version=__version__)

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def index() -> str:
        return resources.files("archivarius.web").joinpath("index.html").read_text("utf-8")

    @app.get("/api/health")
    def health() -> dict[str, object]:
        return {"status": "ok", "version": __version__, "chunks": engine.store.stats()["chunks"]}

    @app.get("/api/stats")
    def stats() -> dict[str, object]:
        return engine.stats()

    @app.get("/api/documents")
    def documents() -> dict[str, object]:
        counts = engine.store.chunk_counts()
        return {
            "documents": [
                {"title": d.title, "path": d.path, "chunks": counts.get(d.id, 0)}
                for d in engine.store.list_documents()
            ]
        }

    @app.get("/api/search")
    def search(
        q: str = Query(min_length=1, max_length=4000),
        k: int | None = Query(default=None, ge=1, le=50),
        mode: str | None = None,
    ) -> dict[str, object]:
        _check_mode(mode)
        try:
            hits = engine.search(q, k, mode)
        except OllamaError as exc:
            raise HTTPException(503, str(exc)) from exc
        return {"query": q, "hits": [hit.to_dict() for hit in hits]}

    @app.post("/api/ask")
    def ask(request: AskRequest):
        _check_mode(request.mode)
        if request.stream:
            return StreamingResponse(
                _ndjson(engine.ask_stream(request.question, request.k, request.mode)),
                media_type="application/x-ndjson",
            )
        try:
            return engine.ask(request.question, request.k, request.mode).to_dict()
        except OllamaError as exc:
            raise HTTPException(503, str(exc)) from exc

    return app


def _ndjson(events: Iterator[dict[str, object]]) -> Iterator[str]:
    try:
        for event in events:
            yield json.dumps(event, ensure_ascii=False) + "\n"
    except OllamaError as exc:
        yield json.dumps({"type": "error", "message": str(exc)}, ensure_ascii=False) + "\n"
