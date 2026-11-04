import shutil
from pathlib import Path

import numpy as np
import pytest

from archivarius.engine import Engine
from archivarius.models import HashingEmbedder
from archivarius.retrieval import reciprocal_rank_fusion
from archivarius.store import EmbeddingMismatchError, IndexStore


def test_index_report(engine: Engine, corpus_dir: Path):
    report = engine.index(corpus_dir)
    assert report.added == 6
    assert report.failed == []
    stats = engine.stats()
    assert stats["documents"] == 6
    assert stats["chunks"] == report.chunks > 6


def test_incremental_indexing(engine: Engine, corpus_dir: Path, tmp_path: Path):
    work = tmp_path / "docs"
    shutil.copytree(corpus_dir, work)
    engine.index(work)

    again = engine.index(work)
    assert (again.added, again.updated, again.unchanged) == (0, 0, 6)

    (work / "git.md").write_text("# Git\nНовое содержимое про ветки.", encoding="utf-8")
    (work / "docker.md").unlink()
    (work / "new.txt").write_text("Свежий документ о метриках.", encoding="utf-8")
    changed = engine.index(work)
    assert (changed.added, changed.updated, changed.removed) == (1, 1, 1)
    assert engine.stats()["documents"] == 6


def test_rebuild_clears_index(indexed_engine: Engine, corpus_dir: Path):
    report = indexed_engine.index(corpus_dir, rebuild=True)
    assert report.added == 6


@pytest.mark.parametrize("mode", ["bm25", "dense", "hybrid"])
def test_search_modes_find_relevant_document(indexed_engine: Engine, mode: str):
    hits = indexed_engine.search("Чем RANK отличается от DENSE_RANK", k=3, mode=mode)
    assert hits
    assert hits[0].chunk.title == "sql-window-functions.md"


def test_hybrid_reports_component_ranks(indexed_engine: Engine):
    hits = indexed_engine.search("docker compose сервисы", k=5, mode="hybrid")
    assert any(h.sparse_rank and h.dense_rank for h in hits)
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)


def test_search_empty_query_and_bad_mode(indexed_engine: Engine):
    assert indexed_engine.search("   ") == []
    with pytest.raises(ValueError):
        indexed_engine.search("git", mode="magic")


def test_search_on_empty_index(engine: Engine):
    assert engine.search("что угодно") == []


def test_rrf_order():
    fused = reciprocal_rank_fusion([[1, 2, 3], [3, 1, 4]], k=60)
    assert fused[0][0] == 1
    assert {item for item, _ in fused} == {1, 2, 3, 4}


def test_embedding_model_mismatch(tmp_path: Path):
    store = IndexStore(tmp_path / "x.db")
    store.check_embedding_model("model-a", 8)
    store.check_embedding_model("model-a", 8)
    with pytest.raises(EmbeddingMismatchError):
        store.check_embedding_model("model-b", 8)
    store.close()


def test_store_roundtrip(tmp_path: Path):
    from archivarius.chunking import Chunk

    store = IndexStore(tmp_path / "x.db")
    chunks = [Chunk("альфа", "стр. 1", 0), Chunk("бета", "стр. 2", 1)]
    vectors = HashingEmbedder(16).embed([c.text for c in chunks])
    store.replace_document("/a.txt", "a.txt", "sha", 10, chunks, vectors)
    records, matrix = store.load_chunks()
    assert [r.text for r in records] == ["альфа", "бета"]
    assert np.allclose(matrix, vectors)
    assert store.delete_document("/a.txt")
    assert store.load_chunks()[0] == []
    store.close()


def test_server_sees_index_changes_from_another_process(tmp_path: Path, corpus_dir: Path):
    from archivarius.config import Settings
    from archivarius.models import EchoChatModel

    settings = Settings(index_path=str(tmp_path / "shared.db"))
    server = Engine(settings, embedder=HashingEmbedder(64), llm=EchoChatModel())
    assert server.search("git revert") == []

    cli = Engine(settings, embedder=HashingEmbedder(64), llm=EchoChatModel())
    cli.index(corpus_dir)
    cli.close()

    assert server.search("git revert")[0].chunk.title == "git.md"
    server.close()


def test_search_limits_chunks_per_document(indexed_engine: Engine):
    from collections import Counter

    for mode in ("bm25", "dense", "hybrid"):
        hits = indexed_engine.search("оконные функции ранжирование смещения рамки", k=6, mode=mode)
        per_doc = Counter(h.chunk.path for h in hits)
        assert max(per_doc.values()) <= 2
        assert len(per_doc) >= 2


def test_per_document_limit_can_be_disabled(indexed_engine: Engine):
    from collections import Counter

    indexed_engine.settings.max_chunks_per_doc = 0
    hits = indexed_engine.search("оконные функции ранжирование смещения рамки", k=5, mode="bm25")
    assert Counter(h.chunk.path for h in hits).most_common(1)[0][1] > 2
