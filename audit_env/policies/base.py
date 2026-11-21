"""Base policy interface for Redline."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BasePolicy(ABC):
    """Minimal agent policy interface compatible with AuditEnv and IsolatedAuditEnv."""

    name: str = "base"

    @abstractmethod
    def reset(self, observation: dict[str, Any], info: dict[str, Any]) -> None:
        """Initialize per-episode policy state from reset() output."""

    @abstractmethod
    def act(self, observation: dict[str, Any]) -> dict[str, Any]:
        """Choose the next JSON action given the latest observation."""
