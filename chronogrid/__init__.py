"""
Chronogrid - Video Processing and AI Analysis Toolkit

A comprehensive toolkit for processing videos into chronological grids
with AI-powered analysis using Llama vision models.
"""

__version__ = "1.0.0"
__author__ = "Chronogrid Team"
__description__ = "Video processing toolkit with AI analysis"

# Main entry points
from chronogrid.core.processing import (
    process_video,
    process_footage,
    ensure_ffmpeg,
    iter_video_files,
    ChronogridResult
)

from chronogrid.core.api_client import LlamaAPIClient

__all__ = [
    "process_video",
    "process_footage",
    "ensure_ffmpeg",
    "iter_video_files",
    "ChronogridResult",
    "LlamaAPIClient",
    "__version__",
]