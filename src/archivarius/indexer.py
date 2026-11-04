"""Инкрементальная индексация каталога с документами.

Для каждого файла считается SHA-256. Неизменённые файлы пропускаются,
изменённые переиндексируются, удалённые с диска убираются из индекса.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from archivarius.chunking import chunk_segments
from archivarius.config import Settings
from archivarius.loaders import iter_documents, load_document
from archivarius.models import Embedder
from archivarius.ollama_client import OllamaError
from archivarius.store import EmbeddingMismatchError, IndexStore

ProgressCallback = Callable[[str, Path], None]


@dataclass
class IndexReport:
    added: int = 0
    updated: int = 0
    unchanged: int = 0
    removed: int = 0
    empty: int = 0
    chunks: int = 0
    failed: list[tuple[str, str]] = field(default_factory=list)

    @property
    def processed(self) -> int:
        return self.added + self.updated + self.unchanged + self.empty + len(self.failed)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_under(path: str, root: Path) -> bool:
    try:
        Path(path).relative_to(root)
        return True
    except ValueError:
        return False


def index_directory(
    root: str | Path,
    store: IndexStore,
    embedder: Embedder,
    settings: Settings,
    progress: ProgressCallback | None = None,
    prune: bool = True,
) -> IndexReport:
    root = Path(root).resolve()
    if not root.exists():
        raise FileNotFoundError(f"Путь не найден: {root}")

    report = IndexReport()
    seen: set[str] = set()

    for path in iter_documents(root):
        key = str(path.resolve())
        seen.add(key)
        try:
            sha = file_sha256(path)
            existing = store.get_document(key)
            if existing and existing.sha256 == sha:
                report.unchanged += 1
                if progress:
                    progress("unchanged", path)
                continue

            document = load_document(path)
            chunks = chunk_segments(document.segments, settings.chunk_size, settings.chunk_overlap)
            if not chunks:
                report.empty += 1
                if existing:
                    store.delete_document(key)
                if progress:
                    progress("empty", path)
                continue

            vectors = embedder.embed([settings.passage_prefix + c.text for c in chunks])
            store.check_embedding_model(embedder.name, int(vectors.shape[1]))
            store.replace_document(key, document.title, sha, path.stat().st_size, chunks, vectors)
            report.chunks += len(chunks)
            if existing:
                report.updated += 1
            else:
                report.added += 1
            if progress:
                progress("updated" if existing else "added", path)
        except (EmbeddingMismatchError, OllamaError):
            raise
        except Exception as exc:  # noqa: BLE001 - один битый файл не должен ронять индексацию
            report.failed.append((str(path), f"{exc.__class__.__name__}: {exc}"))
            if progress:
                progress("failed", path)

    if prune:
        for document in store.list_documents():
            if _is_under(document.path, root) and document.path not in seen:
                store.delete_document(document.path)
                report.removed += 1
                if progress:
                    progress("removed", Path(document.path))
    return report
