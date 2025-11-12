"""Exception types shared by the Chronogrid pipeline."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class ChronogridPipelineError(RuntimeError):
    """Raised when a pipeline stage cannot complete successfully."""

    message: str
    stage: str | None = None

    def __post_init__(self) -> None:
        super().__init__(self.formatted_message)

    @property
    def formatted_message(self) -> str:
        if self.stage:
            return f"{self.stage}: {self.message}"
        return self.message


__all__ = ["ChronogridPipelineError"]
