"""Input validation for the Chronogrid pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from .errors import ChronogridPipelineError
from .models import ValidationResult

SUPPORTED_EXTENSIONS = {".mp4", ".mov", ".m4v", ".avi", ".mkv"}
DEFAULT_OUTPUT_DIR = Path("outputs")


def _find_media_files(paths: Sequence[str]) -> List[Path]:
    resolved: List[Path] = []
    candidates = paths or ["."]

    for target in candidates:
        path = Path(target).expanduser().resolve()
        if not path.exists():
            raise ChronogridPipelineError(f"Input path does not exist: {path}", stage="input_validation")

        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            resolved.append(path)
            continue

        if path.is_dir():
            matches = [
                candidate
                for candidate in path.rglob("*")
                if candidate.is_file() and candidate.suffix.lower() in SUPPORTED_EXTENSIONS
            ]
            if not matches:
                raise ChronogridPipelineError(
                    f"No supported videos found inside directory: {path}",
                    stage="input_validation",
                )
            resolved.extend(sorted(matches))
            continue

        raise ChronogridPipelineError(
            f"Unsupported input path (not a file or directory): {path}",
            stage="input_validation",
        )

    if not resolved:
        raise ChronogridPipelineError("No video files were discovered", stage="input_validation")

    # Deduplicate while maintaining order
    deduped: List[Path] = []
    seen = set()
    for item in resolved:
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped


def _validate_options(options: Dict[str, Any] | None) -> Dict[str, Any]:
    opts: Dict[str, Any] = dict(options or {})

    frame_step = int(opts.get("frame_step", 30))
    if frame_step < 1:
        raise ChronogridPipelineError("frame_step must be >= 1", stage="input_validation")

    grid_size = int(opts.get("grid_size", 4))
    if not 2 <= grid_size <= 10:
        raise ChronogridPipelineError("grid_size must be between 2 and 10", stage="input_validation")

    analyze = bool(opts.get("analyze", True))
    cleanup = bool(opts.get("cleanup", True))
    output_dir = Path(opts.get("output_dir", DEFAULT_OUTPUT_DIR)).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    prompt = opts.get("prompt")
    if prompt is not None and not isinstance(prompt, str):
        raise ChronogridPipelineError("prompt must be a string when provided", stage="input_validation")

    opts.update(
        {
            "frame_step": frame_step,
            "grid_size": grid_size,
            "analyze": analyze,
            "cleanup": cleanup,
            "output_dir": output_dir,
            "prompt": prompt,
        }
    )
    return opts


def validate_inputs(paths: Sequence[str], options: Dict[str, Any] | None = None) -> ValidationResult:
    """Validate and normalize CLI inputs."""
    videos = _find_media_files(list(paths))
    normalized_options = _validate_options(options)
    return ValidationResult(video_paths=videos, options=normalized_options)


__all__ = ["validate_inputs", "SUPPORTED_EXTENSIONS", "DEFAULT_OUTPUT_DIR"]
