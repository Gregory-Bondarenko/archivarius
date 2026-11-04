"""Настройки приложения.

Источники в порядке приоритета: переменные окружения ARCHIVARIUS_*,
YAML-файл конфигурации, значения по умолчанию.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import yaml

ENV_PREFIX = "ARCHIVARIUS_"
DEFAULT_CONFIG_NAME = "archivarius.yaml"
SEARCH_MODES = ("bm25", "dense", "hybrid")


@dataclass
class Settings:
    # Ollama
    ollama_url: str = "http://127.0.0.1:11434"
    embed_model: str = "bge-m3"
    llm_model: str = "qwen3:8b"
    request_timeout: float = 600.0
    temperature: float = 0.1
    num_ctx: int = 8192
    think: bool = False

    # Индекс
    index_path: str = ".archivarius/index.db"
    chunk_size: int = 1000
    chunk_overlap: int = 150
    embed_batch_size: int = 32
    query_prefix: str = ""
    passage_prefix: str = ""

    # Поиск
    top_k: int = 5
    search_mode: str = "hybrid"
    rrf_k: int = 60
    candidate_pool: int = 10
    max_chunks_per_doc: int = 2
    bm25_k1: float = 1.5
    bm25_b: float = 0.75

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if self.search_mode not in SEARCH_MODES:
            raise ValueError(f"search_mode должен быть одним из {SEARCH_MODES}")
        if self.chunk_size < 100:
            raise ValueError("chunk_size должен быть не меньше 100 символов")
        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError("chunk_overlap должен быть в диапазоне [0, chunk_size)")
        if self.max_chunks_per_doc < 0:
            raise ValueError("max_chunks_per_doc не может быть отрицательным")
        if self.top_k < 1:
            raise ValueError("top_k должен быть положительным")

    @classmethod
    def load(cls, path: str | Path | None = None) -> Settings:
        """Собирает настройки из файла и окружения."""
        values: dict[str, Any] = {}
        config_path = Path(path) if path else Path(DEFAULT_CONFIG_NAME)
        if config_path.is_file():
            data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
            if not isinstance(data, dict):
                raise ValueError(f"Некорректный формат конфигурации: {config_path}")
            values.update(data)
        elif path:
            raise FileNotFoundError(f"Файл конфигурации не найден: {config_path}")

        known = {f.name: f for f in fields(cls)}
        unknown = set(values) - set(known)
        if unknown:
            raise ValueError(f"Неизвестные параметры конфигурации: {sorted(unknown)}")

        for name, field in known.items():
            raw = os.environ.get(ENV_PREFIX + name.upper())
            if raw is not None:
                values[name] = _cast(raw, field.type)
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _cast(raw: str, type_name: Any) -> Any:
    type_name = str(type_name)
    if type_name == "bool":
        return raw.strip().lower() in {"1", "true", "yes", "on", "да"}
    if type_name == "int":
        return int(raw)
    if type_name == "float":
        return float(raw)
    return raw
