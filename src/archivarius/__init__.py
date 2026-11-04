"""Archivarius: локальный вопросно-ответный поиск по документам."""

__version__ = "0.1.0"

from archivarius.config import Settings
from archivarius.engine import Engine

__all__ = ["Engine", "Settings", "__version__"]
