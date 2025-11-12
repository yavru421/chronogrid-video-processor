"""
Chronogrid - Video Processing and AI Analysis Toolkit

A comprehensive toolkit for processing videos into chronological grids
with AI-powered analysis using Llama vision models.
"""

__version__ = "1.1.0"
__author__ = "Chronogrid Team"
__description__ = "Video processing toolkit with AI analysis"

# Main entry points
from chronogrid.core.processing import (
    process_video,
    ensure_ffmpeg,
    iter_video_files,
    ChronogridResult,
    create_semantic_summary,
    save_semantic_summary,
)
from chronogrid.core.api_client import LlamaAPIClient
from chronogrid.core.licensing import (
    has_valid_license,
    get_usage_count,
    activate_license,
    get_upgrade_message,
)
from chronogrid.pipeline import ChronogridPipeline, run_pipeline

__all__ = [
    "process_video",
    "ensure_ffmpeg",
    "iter_video_files",
    "ChronogridResult",
    "create_semantic_summary",
    "save_semantic_summary",
    "LlamaAPIClient",
    "has_valid_license",
    "get_usage_count",
    "activate_license",
    "get_upgrade_message",
    "ChronogridPipeline",
    "run_pipeline",
    "__version__",
]
