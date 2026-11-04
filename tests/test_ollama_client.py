import json

import httpx
import numpy as np
import pytest

from archivarius.models import OllamaChatModel, OllamaEmbedder
from archivarius.ollama_client import OllamaClient, OllamaError, is_local_url


def fake_ollama(request: httpx.Request) -> httpx.Response:
    path = request.url.path
    if path == "/api/version":
        return httpx.Response(200, json={"version": "0.12.0"})
    if path == "/api/tags":
        return httpx.Response(200, json={"models": [{"name": "bge-m3:latest"}]})
    body = json.loads(request.content)
    if body.get("model") == "missing":
        return httpx.Response(404, json={"error": "model not found"})
    if path == "/api/show":
        caps = ["completion", "thinking"] if "qwen3" in body["model"] else ["completion"]
        return httpx.Response(200, json={"capabilities": caps})
    if path == "/api/embed":
        return httpx.Response(200, json={"embeddings": [[3.0, 4.0] for _ in body["input"]]})
    if path == "/api/chat":
        assert body["think"] is False
        if body["stream"]:
            lines = [
                {"message": {"content": "Отв"}, "done": False},
                {"message": {"content": "ет [1]"}, "done": False},
                {"message": {"content": ""}, "done": True},
            ]
            content = "\n".join(json.dumps(x, ensure_ascii=False) for x in lines)
            return httpx.Response(200, content=content.encode("utf-8"))
        return httpx.Response(200, json={"message": {"content": "Ответ [1]"}, "done": True})
    return httpx.Response(404)


@pytest.fixture
def client() -> OllamaClient:
    return OllamaClient(transport=httpx.MockTransport(fake_ollama))


def test_version_and_models(client: OllamaClient):
    assert client.version() == "0.12.0"
    assert client.list_models() == ["bge-m3:latest"]


def test_embedder_batches_and_normalizes(client: OllamaClient):
    embedder = OllamaEmbedder(client, "bge-m3", batch_size=2)
    vectors = embedder.embed(["a", "b", "c"])
    assert vectors.shape == (3, 2)
    assert np.allclose(vectors[0], [0.6, 0.8])


def test_chat_and_stream(client: OllamaClient):
    model = OllamaChatModel(client, "qwen3:8b")
    assert model.complete([{"role": "user", "content": "?"}]) == "Ответ [1]"
    assert "".join(model.stream([{"role": "user", "content": "?"}])) == "Ответ [1]"


def test_missing_model_message(client: OllamaClient):
    with pytest.raises(OllamaError, match="ollama pull missing"):
        client.embed("missing", ["x"])
    with pytest.raises(OllamaError, match="ollama pull missing"):
        list(client.chat_stream("missing", [{"role": "user", "content": "?"}]))


def test_local_url_ignores_system_proxy(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("HTTP_PROXY", "http://proxy.invalid:3128")
    assert is_local_url("http://127.0.0.1:11434")
    assert is_local_url("http://localhost:11434")
    assert not is_local_url("http://gpu-server:11434")
    client = OllamaClient("http://127.0.0.1:11434")
    assert client._client._trust_env is False
    client.close()


def test_unreachable_server_message():
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    client = OllamaClient(transport=httpx.MockTransport(refuse))
    with pytest.raises(OllamaError, match="ConnectError"):
        client.version()


def test_read_timeout_message():
    def slow(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    client = OllamaClient(timeout=5, transport=httpx.MockTransport(slow))
    with pytest.raises(OllamaError, match="не ответила за 5"):
        client.chat("qwen3:14b", [{"role": "user", "content": "?"}])


def test_reasoning_capability(client: OllamaClient):
    assert OllamaChatModel(client, "qwen3:4b").reasoning is True
    assert OllamaChatModel(client, "gemma3:4b").reasoning is False
