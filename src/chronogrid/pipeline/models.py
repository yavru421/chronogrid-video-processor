"""Shared dataclasses for Chronogrid pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from chronogrid.core.processing import ChronogridResult


@dataclass(slots=True)
class ValidationResult:
    video_paths: List[Path]
    options: Dict[str, Any]


@dataclass(slots=True)
class DependencyResult:
    ffmpeg_path: Path
    python_deps: Dict[str, bool]
    environment: Dict[str, str]


@dataclass(slots=True)
class ProcessingResult:
    artifacts: List[ChronogridResult]
    options: Dict[str, Any]
    duration_seconds: Optional[float] = None


@dataclass(slots=True)
class OutputArtifact:
    video_path: Path
    chronogrid_path: Path
    analysis_path: Optional[Path]
    metadata_path: Path
    analysis_text: str | None = None


@dataclass(slots=True)
class OutputResult:
    base_dir: Path
    files: List[OutputArtifact] = field(default_factory=list)
    summary_path: Path | None = None
    metadata: Dict[str, Any] = field(default_factory=dict)


__all__ = [
    "ValidationResult",
    "DependencyResult",
    "ProcessingResult",
    "OutputArtifact",
    "OutputResult",
]
