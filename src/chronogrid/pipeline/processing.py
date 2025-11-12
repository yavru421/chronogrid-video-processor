"""Processing stage – drives the heavy lifting via chronogrid.core."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

from chronogrid.core.processing import ChronogridResult, process_video

from .errors import ChronogridPipelineError
from .models import DependencyResult, ProcessingResult, ValidationResult


def _analysis_prompt(options: dict[str, object]) -> Optional[str]:
    if not options.get("analyze", True):
        return None
    prompt = options.get("prompt")
    return str(prompt) if prompt else None


def execute_processing(validation_result: ValidationResult, _: DependencyResult | None = None) -> ProcessingResult:
    """Run chronogrid.core.process_video for each validated input."""
    start = time.perf_counter()
    artifacts: list[ChronogridResult] = []
    options = validation_result.options
    prompt = _analysis_prompt(options)

    for video_path in validation_result.video_paths:
        try:
            artifact = process_video(
                video_path,
                frame_step=options["frame_step"],
                grid_size=options["grid_size"],
                prompt=prompt,
                cleanup=options["cleanup"],
            )
            artifacts.append(artifact)
        except Exception as exc:  # pragma: no cover - adapters wrap arbitrary ffmpeg errors
            raise ChronogridPipelineError(
                f"Failed while processing {Path(video_path).name}: {exc}",
                stage="processing",
            ) from exc

    duration = round(time.perf_counter() - start, 3)
    return ProcessingResult(artifacts=artifacts, options=options, duration_seconds=duration)


__all__ = ["execute_processing"]
