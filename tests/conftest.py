from pathlib import Path

import pytest

from archivarius.config import Settings
from archivarius.engine import Engine
from archivarius.models import EchoChatModel, HashingEmbedder
from archivarius.store import IndexStore

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


@pytest.fixture
def corpus_dir() -> Path:
    return EXAMPLES / "corpus"


@pytest.fixture
def eval_path() -> Path:
    return EXAMPLES / "eval.jsonl"


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(index_path=str(tmp_path / "index.db"), chunk_size=400, chunk_overlap=80)


@pytest.fixture
def engine(settings: Settings) -> Engine:
    eng = Engine(
        settings,
        embedder=HashingEmbedder(256),
        llm=EchoChatModel(),
        store=IndexStore(settings.index_path),
    )
    yield eng
    eng.close()


@pytest.fixture
def indexed_engine(engine: Engine, corpus_dir: Path) -> Engine:
    engine.index(corpus_dir)
    return engine
