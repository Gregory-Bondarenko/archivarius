"""Минимальный клиент REST API Ollama: эмбеддинги, чат, список моделей."""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any
from urllib.parse import urlparse

import httpx

LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", "0.0.0.0"}


class OllamaError(RuntimeError):
    pass


class OllamaClient:
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:11434",
        timeout: float = 600.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        # Для локальной Ollama системный прокси (VPN, корпоративная сеть) только мешает
        self._client = httpx.Client(
            base_url=self.base_url,
            timeout=httpx.Timeout(timeout, connect=10.0),
            transport=transport,
            trust_env=not is_local_url(self.base_url),
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> OllamaClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _unavailable(self, exc: Exception) -> OllamaError:
        if isinstance(exc, httpx.ReadTimeout):
            return OllamaError(
                f"Модель не ответила за {self.timeout:.0f} с. При первом запросе Ollama загружает "
                "модель в память, это может занять несколько минут. Повторите запрос или увеличьте "
                "request_timeout. Команда `ollama ps` покажет, считается ли модель на видеокарте."
            )
        return OllamaError(
            f"Ollama недоступна по адресу {self.base_url} ({exc.__class__.__name__}: {exc}). "
            "Проверьте, что Ollama запущена, и адрес ollama_url в настройках."
        )

    def _get(self, path: str) -> dict[str, Any]:
        try:
            response = self._client.get(path)
        except httpx.TransportError as exc:
            raise self._unavailable(exc) from exc
        if response.is_error:
            detail = response.text[:300]
            raise OllamaError(f"Ошибка Ollama {response.status_code} на {path}: {detail}")
        return response.json()

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            response = self._client.post(path, json=payload)
        except httpx.TransportError as exc:
            raise self._unavailable(exc) from exc
        if response.status_code == 404:
            model = payload.get("model", "")
            raise OllamaError(f"Модель «{model}» не найдена. Выполните `ollama pull {model}`.")
        if response.is_error:
            raise OllamaError(f"Ошибка Ollama {response.status_code}: {response.text[:300]}")
        return response.json()

    def list_models(self) -> list[str]:
        return [m["name"] for m in self._get("/api/tags").get("models", [])]

    def version(self) -> str:
        return self._get("/api/version").get("version", "unknown")

    def capabilities(self, model: str) -> list[str]:
        """Возможности модели по данным Ollama: completion, thinking, embedding и т. д."""
        return list(self._post("/api/show", {"model": model}).get("capabilities") or [])

    def embed(self, model: str, texts: list[str]) -> list[list[float]]:
        data = self._post("/api/embed", {"model": model, "input": texts})
        embeddings = data.get("embeddings")
        if not embeddings or len(embeddings) != len(texts):
            raise OllamaError("Ollama вернула некорректный ответ на запрос эмбеддингов")
        return embeddings

    def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        options: dict[str, Any] | None = None,
        think: bool | None = None,
    ) -> str:
        payload: dict[str, Any] = {"model": model, "messages": messages, "stream": False}
        if options:
            payload["options"] = options
        if think is not None:
            payload["think"] = think
        data = self._post("/api/chat", payload)
        return data.get("message", {}).get("content", "")

    def chat_stream(
        self,
        model: str,
        messages: list[dict[str, str]],
        options: dict[str, Any] | None = None,
        think: bool | None = None,
    ) -> Iterator[str]:
        payload: dict[str, Any] = {"model": model, "messages": messages, "stream": True}
        if options:
            payload["options"] = options
        if think is not None:
            payload["think"] = think
        try:
            with self._client.stream("POST", "/api/chat", json=payload) as response:
                if response.status_code == 404:
                    raise OllamaError(
                        f"Модель «{model}» не найдена. Выполните `ollama pull {model}`."
                    )
                if response.is_error:
                    response.read()
                    raise OllamaError(f"Ошибка Ollama {response.status_code}: {response.text}")
                for line in response.iter_lines():
                    if not line:
                        continue
                    event = json.loads(line)
                    if "error" in event:
                        raise OllamaError(event["error"])
                    token = event.get("message", {}).get("content", "")
                    if token:
                        yield token
                    if event.get("done"):
                        break
        except httpx.TransportError as exc:
            raise self._unavailable(exc) from exc


def is_local_url(url: str) -> bool:
    return (urlparse(url).hostname or "").lower() in LOCAL_HOSTS
