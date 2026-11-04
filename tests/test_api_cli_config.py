import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from archivarius.api import create_app
from archivarius.cli import app as cli_app
from archivarius.config import Settings
from archivarius.engine import Engine


@pytest.fixture
def client(indexed_engine: Engine) -> TestClient:
    return TestClient(create_app(indexed_engine))


def test_health_and_ui(client: TestClient):
    assert client.get("/api/health").json()["status"] == "ok"
    assert "Archivarius" in client.get("/").text


def test_documents_endpoint(client: TestClient):
    docs = client.get("/api/documents").json()["documents"]
    assert len(docs) == 6
    assert all(d["chunks"] > 0 for d in docs)


def test_search_endpoint(client: TestClient):
    data = client.get("/api/search", params={"q": "оконные функции", "k": 2}).json()
    assert len(data["hits"]) == 2
    assert client.get("/api/search", params={"q": "x", "mode": "bad"}).status_code == 422


def test_ask_endpoint(client: TestClient):
    data = client.post("/api/ask", json={"question": "Как создать окружение?"}).json()
    assert data["answer"]
    assert data["sources"]


def test_ask_stream_endpoint(client: TestClient):
    response = client.post("/api/ask", json={"question": "Что такое MDE?", "stream": True})
    events = [json.loads(line) for line in response.text.splitlines() if line]
    assert [events[0]["type"], events[-1]["type"]] == ["sources", "done"]


def test_settings_env_and_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    config = tmp_path / "c.yaml"
    config.write_text("top_k: 7\nsearch_mode: bm25\n", encoding="utf-8")
    monkeypatch.setenv("ARCHIVARIUS_TOP_K", "9")
    monkeypatch.setenv("ARCHIVARIUS_THINK", "true")
    settings = Settings.load(config)
    assert settings.top_k == 9
    assert settings.search_mode == "bm25"
    assert settings.think is True


def test_settings_validation(tmp_path: Path):
    with pytest.raises(ValueError):
        Settings(search_mode="magic")
    with pytest.raises(ValueError):
        Settings(chunk_size=500, chunk_overlap=500)
    bad = tmp_path / "bad.yaml"
    bad.write_text("unknown_option: 1\n", encoding="utf-8")
    with pytest.raises(ValueError):
        Settings.load(bad)


def test_cli_init_and_version(tmp_path: Path):
    runner = CliRunner()
    target = tmp_path / "archivarius.yaml"
    result = runner.invoke(cli_app, ["init", str(target)])
    assert result.exit_code == 0
    assert Settings.load(target).embed_model == "bge-m3"
    assert "archivarius" in runner.invoke(cli_app, ["version"]).output


def test_serve_reports_busy_port():
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as busy:
        busy.bind(("127.0.0.1", 0))
        busy.listen(1)
        port = busy.getsockname()[1]
        result = CliRunner().invoke(cli_app, ["serve", "--port", str(port)])
    assert result.exit_code == 1
    assert "занят" in result.output
