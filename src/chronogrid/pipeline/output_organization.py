"""Organize chronogrid outputs and emit structured metadata."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import List

from .errors import ChronogridPipelineError
from .models import OutputArtifact, OutputResult, ProcessingResult, ValidationResult


def _write_analysis(chronogrid_path: Path, analysis_text: str | None) -> Path | None:
    if not analysis_text:
        return None
    analysis_path = chronogrid_path.with_name(f"{chronogrid_path.stem}_analysis.txt")
    analysis_path.write_text(analysis_text.strip() + "\n", encoding="utf-8")
    return analysis_path


def _write_metadata(video_path: Path, chronogrid_path: Path, analysis_path: Path | None, options: dict) -> Path:
    metadata = {
        "video_path": str(video_path),
        "chronogrid_path": str(chronogrid_path),
        "analysis_path": str(analysis_path) if analysis_path else None,
        "frame_step": options["frame_step"],
        "grid_size": options["grid_size"],
        "analyze": options["analyze"],
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }
    metadata_path = chronogrid_path.with_name(f"{chronogrid_path.stem}_metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata_path


def organize_outputs(processing_result: ProcessingResult, validation_result: ValidationResult) -> OutputResult:
    base_dir = validation_result.options["output_dir"]
    files: List[OutputArtifact] = []

    for artifact in processing_result.artifacts:
        chronogrid_path = Path(artifact.chronogrid_path)
        if not chronogrid_path.exists():
            raise ChronogridPipelineError(
                f"Expected chronogrid missing: {chronogrid_path}",
                stage="output_organization",
            )

        analysis_path = _write_analysis(chronogrid_path, artifact.analysis_text)
        metadata_path = _write_metadata(
            artifact.video_path,
            chronogrid_path,
            analysis_path,
            processing_result.options,
        )
        files.append(
            OutputArtifact(
                video_path=artifact.video_path,
                chronogrid_path=chronogrid_path,
                analysis_path=analysis_path,
                metadata_path=metadata_path,
                analysis_text=artifact.analysis_text,
            )
        )

    summary = {
        "videos_processed": len(files),
        "duration_seconds": processing_result.duration_seconds,
        "options": {
            "frame_step": processing_result.options["frame_step"],
            "grid_size": processing_result.options["grid_size"],
            "analyze": processing_result.options["analyze"],
        },
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }
    summary_path = base_dir / "processing_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return OutputResult(base_dir=base_dir, files=files, summary_path=summary_path, metadata=summary)


__all__ = ["organize_outputs"]
