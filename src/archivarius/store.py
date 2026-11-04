"""Хранилище индекса в одном файле SQLite.

Документы, фрагменты и их векторы лежат в одной базе, что упрощает
перенос индекса и делает обновление атомарным на уровне документа.
Векторы хранятся как BLOB float32 и при чтении собираются в матрицу NumPy.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from archivarius.chunking import Chunk

SCHEMA_VERSION = "1"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS documents (
    id         INTEGER PRIMARY KEY,
    path       TEXT NOT NULL UNIQUE,
    title      TEXT NOT NULL,
    sha256     TEXT NOT NULL,
    size       INTEGER NOT NULL,
    indexed_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS chunks (
    id        INTEGER PRIMARY KEY,
    doc_id    INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    ord       INTEGER NOT NULL,
    location  TEXT NOT NULL,
    text      TEXT NOT NULL,
    embedding BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
"""


@dataclass(frozen=True)
class DocumentRecord:
    id: int
    path: str
    title: str
    sha256: str
    size: int
    indexed_at: str


@dataclass(frozen=True)
class ChunkRecord:
    id: int
    doc_id: int
    ord: int
    location: str
    text: str
    path: str
    title: str


class EmbeddingMismatchError(RuntimeError):
    pass


class IndexStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if str(path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path), check_same_thread=False)
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.execute("PRAGMA journal_mode = WAL")
        self.conn.executescript(_SCHEMA)
        if self.get_meta("schema_version") is None:
            self.set_meta("schema_version", SCHEMA_VERSION)
        self._local_revision = 0

    def close(self) -> None:
        self.conn.close()

    @property
    def revision(self) -> tuple[int, int]:
        """Версия данных. Меняется и при своих записях, и при записях из другого процесса.

        PRAGMA data_version растёт, когда базу изменило другое соединение, поэтому
        запущенный веб-сервер видит результат `archivarius index` без перезапуска.
        """
        data_version = self.conn.execute("PRAGMA data_version").fetchone()[0]
        return self._local_revision, int(data_version)

    # --- служебные данные -------------------------------------------------

    def get_meta(self, key: str) -> str | None:
        row = self.conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None

    def set_meta(self, key: str, value: str) -> None:
        with self.conn:
            self.conn.execute(
                "INSERT INTO meta(key, value) VALUES(?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, value),
            )

    def check_embedding_model(self, model: str, dim: int) -> None:
        """Не позволяет смешать в одном индексе векторы разных моделей."""
        stored_model = self.get_meta("embed_model")
        stored_dim = self.get_meta("embed_dim")
        if stored_model is None:
            self.set_meta("embed_model", model)
            self.set_meta("embed_dim", str(dim))
            return
        if stored_model != model or int(stored_dim or 0) != dim:
            raise EmbeddingMismatchError(
                f"Индекс построен моделью «{stored_model}» (размерность {stored_dim}), "
                f"а сейчас используется «{model}» ({dim}). Пересоберите индекс с флагом --rebuild."
            )

    # --- документы --------------------------------------------------------

    def get_document(self, path: str) -> DocumentRecord | None:
        row = self.conn.execute(
            "SELECT id, path, title, sha256, size, indexed_at FROM documents WHERE path = ?",
            (path,),
        ).fetchone()
        return DocumentRecord(*row) if row else None

    def list_documents(self) -> list[DocumentRecord]:
        rows = self.conn.execute(
            "SELECT id, path, title, sha256, size, indexed_at FROM documents ORDER BY path"
        ).fetchall()
        return [DocumentRecord(*row) for row in rows]

    def replace_document(
        self,
        path: str,
        title: str,
        sha256: str,
        size: int,
        chunks: list[Chunk],
        embeddings: np.ndarray,
    ) -> int:
        """Атомарно заменяет документ и все его фрагменты."""
        if len(chunks) != len(embeddings):
            raise ValueError("Число фрагментов и векторов не совпадает")
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        vectors = np.asarray(embeddings, dtype=np.float32)
        with self.conn:
            self.conn.execute("DELETE FROM documents WHERE path = ?", (path,))
            cursor = self.conn.execute(
                "INSERT INTO documents(path, title, sha256, size, indexed_at) VALUES(?,?,?,?,?)",
                (path, title, sha256, size, now),
            )
            doc_id = int(cursor.lastrowid)
            self.conn.executemany(
                "INSERT INTO chunks(doc_id, ord, location, text, embedding) VALUES(?,?,?,?,?)",
                [
                    (doc_id, c.ord, c.location, c.text, vectors[i].tobytes())
                    for i, c in enumerate(chunks)
                ],
            )
        self._local_revision += 1
        return doc_id

    def delete_document(self, path: str) -> bool:
        with self.conn:
            cursor = self.conn.execute("DELETE FROM documents WHERE path = ?", (path,))
        if cursor.rowcount:
            self._local_revision += 1
        return bool(cursor.rowcount)

    def clear(self) -> None:
        with self.conn:
            self.conn.execute("DELETE FROM documents")
            self.conn.execute("DELETE FROM meta WHERE key IN ('embed_model', 'embed_dim')")
        self._local_revision += 1

    # --- фрагменты --------------------------------------------------------

    def load_chunks(self) -> tuple[list[ChunkRecord], np.ndarray]:
        rows = self.conn.execute(
            "SELECT c.id, c.doc_id, c.ord, c.location, c.text, d.path, d.title, c.embedding "
            "FROM chunks c JOIN documents d ON d.id = c.doc_id ORDER BY c.doc_id, c.ord"
        ).fetchall()
        records = [ChunkRecord(*row[:7]) for row in rows]
        if not rows:
            return records, np.zeros((0, 0), dtype=np.float32)
        matrix = np.vstack([np.frombuffer(row[7], dtype=np.float32) for row in rows])
        return records, matrix

    def chunk_counts(self) -> dict[int, int]:
        rows = self.conn.execute("SELECT doc_id, COUNT(*) FROM chunks GROUP BY doc_id").fetchall()
        return {int(doc_id): int(n) for doc_id, n in rows}

    def stats(self) -> dict[str, object]:
        documents = self.conn.execute("SELECT COUNT(*), COALESCE(SUM(size), 0) FROM documents")
        n_docs, total_bytes = documents.fetchone()
        n_chunks, n_chars = self.conn.execute(
            "SELECT COUNT(*), COALESCE(SUM(LENGTH(text)), 0) FROM chunks"
        ).fetchone()
        return {
            "documents": n_docs,
            "chunks": n_chunks,
            "characters": n_chars,
            "source_bytes": total_bytes,
            "embed_model": self.get_meta("embed_model"),
            "embed_dim": self.get_meta("embed_dim"),
            "index_path": str(self.path),
        }
