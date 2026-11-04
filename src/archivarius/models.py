"""Абстракции моделей: векторизатор текста и диалоговая языковая модель.

Реальные реализации работают через Ollama. Детерминированные заглушки
нужны для тестов и для проверки конвейера без запущенной Ollama.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from typing import Protocol

import numpy as np

from archivarius.ollama_client import OllamaClient
from archivarius.text import tokenize


class Embedder(Protocol):
    name: str

    def embed(self, texts: list[str]) -> np.ndarray: ...


class ChatModel(Protocol):
    name: str

    def complete(self, messages: list[dict[str, str]]) -> str: ...

    def stream(self, messages: list[dict[str, str]]) -> Iterator[str]: ...


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix[None, :]
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


class OllamaEmbedder:
    def __init__(self, client: OllamaClient, model: str, batch_size: int = 32) -> None:
        self.client = client
        self.name = model
        self.batch_size = batch_size

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            vectors.extend(self.client.embed(self.name, texts[start : start + self.batch_size]))
        return l2_normalize(np.array(vectors, dtype=np.float32))


class OllamaChatModel:
    def __init__(
        self,
        client: OllamaClient,
        model: str,
        temperature: float = 0.1,
        num_ctx: int = 8192,
        think: bool = False,
    ) -> None:
        self.client = client
        self.name = model
        self.options = {"temperature": temperature, "num_ctx": num_ctx}
        self.think = think
        self._reasoning: bool | None = None

    @property
    def reasoning(self) -> bool:
        """Умеет ли модель рассуждать. Такие модели могут прислать блок рассуждений в ответе."""
        if self._reasoning is None:
            try:
                self._reasoning = "thinking" in self.client.capabilities(self.name)
            except Exception:  # noqa: BLE001 - старые версии Ollama не отдают capabilities
                self._reasoning = True
        return self._reasoning

    def complete(self, messages: list[dict[str, str]]) -> str:
        return self.client.chat(self.name, messages, self.options, think=self.think)

    def stream(self, messages: list[dict[str, str]]) -> Iterator[str]:
        yield from self.client.chat_stream(self.name, messages, self.options, think=self.think)


class HashingEmbedder:
    """Векторизатор на хешировании стемов: без сети, детерминированный."""

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim
        self.name = f"hashing-{dim}"

    def _index(self, token: str) -> tuple[int, float]:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "little")
        return value % self.dim, 1.0 if (value >> 63) & 1 else -1.0

    def embed(self, texts: list[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in tokenize(text):
                index, sign = self._index(token)
                matrix[row, index] += sign
        return l2_normalize(matrix) if len(texts) else matrix


class EchoChatModel:
    """Заглушка LLM: возвращает первый источник из контекста с цитатой."""

    name = "echo"

    def complete(self, messages: list[dict[str, str]]) -> str:
        return "".join(self.stream(messages))

    def stream(self, messages: list[dict[str, str]]) -> Iterator[str]:
        content = messages[-1]["content"]
        marker = "[1]"
        if marker in content:
            body = content.split(marker, 1)[1].split("\n", 2)[-1]
            answer = body.strip().split("\n")[0][:200]
            text = f"{answer} [1]"
        else:
            text = "В предоставленных документах нет ответа на этот вопрос."
        for word in text.split(" "):
            yield word + " "
