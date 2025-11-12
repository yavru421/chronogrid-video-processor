"""High-level pipeline orchestration for Chronogrid."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Sequence

from .dependency_resolution import resolve_dependencies
from .errors import ChronogridPipelineError
from .input_validation import validate_inputs
from .models import OutputResult
from .output_organization import organize_outputs
from .processing import execute_processing


@dataclass
class ChronogridPipeline:
    """Coordinates validation, dependency checks, processing, and output organization."""

    require_analysis: bool | None = None

    def run(self, inputs: Sequence[str], **options: Any) -> OutputResult:
        validation = validate_inputs(inputs, options)
        analyze = validation.options["analyze"]
        deps = resolve_dependencies(require_analysis=self.require_analysis if self.require_analysis is not None else analyze)
        processing = execute_processing(validation, deps)
        return organize_outputs(processing, validation)


def run_pipeline(inputs: Sequence[str], **options: Any) -> OutputResult:
    """Convenience wrapper mirroring historical API surface."""
    pipeline = ChronogridPipeline()
    return pipeline.run(inputs, **options)


__all__ = ["ChronogridPipeline", "run_pipeline", "ChronogridPipelineError", "OutputResult"]
