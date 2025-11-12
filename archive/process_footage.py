"""Compatibility shim re-exporting processing utilities at top-level.

Legacy tests and scripts import from `process_footage`; this module forwards
to `chronogrid.core.processing` without altering behavior.
"""

from chronogrid.core import processing as _core
from chronogrid.core.processing import (  # explicit re-export for compatibility
	ChronogridResult,
	check_ffmpeg,
	ensure_ffmpeg,
	iter_video_files,
	list_video_files,
	get_video_metadata,
	detect_scenes,
	detect_hardware_acceleration,
	get_cpu_count,
	get_optimal_ffmpeg_args,
	list_extracted_frames,
	extract_frames,
	generate_chronogrid,
	generate_audio_spectrogram,
	get_analysis_prompt,
	parse_analysis_response,
	parse_chronogrid_analysis,
	parse_segmentation_analysis,
	parse_metadata_analysis,
	parse_quality_analysis,
	parse_summary_analysis,
	process_video,
)

import logging as _logging
logger = _logging.getLogger(__name__)

__all__ = [
	"ChronogridResult",
	"check_ffmpeg",
	"ensure_ffmpeg",
	"iter_video_files",
	"list_video_files",
	"get_video_metadata",
	"detect_scenes",
	"detect_hardware_acceleration",
	"get_cpu_count",
	"get_optimal_ffmpeg_args",
	"list_extracted_frames",
	"extract_frames",
	"generate_chronogrid",
	"generate_audio_spectrogram",
	"get_analysis_prompt",
	"parse_analysis_response",
	"parse_chronogrid_analysis",
	"parse_segmentation_analysis",
	"parse_metadata_analysis",
	"parse_quality_analysis",
	"parse_summary_analysis",
	"process_video",
]

from contextlib import contextmanager as _contextmanager


@_contextmanager
def _override_core_dependencies():
	"""Temporarily point core.processing internals at our module-level symbols.

	This allows unit tests that patch process_footage.* to affect the underlying
	core implementation without rewriting the core module.
	"""
	names = [
		"get_video_metadata",
		"detect_scenes",
		"list_extracted_frames",
		"generate_chronogrid",
		"generate_audio_spectrogram",
		"check_ffmpeg",
		"detect_hardware_acceleration",
		"get_cpu_count",
		"get_optimal_ffmpeg_args",
		"logger",
	]
	saved = {n: getattr(_core, n) for n in names}
	try:
		for n in names:
			setattr(_core, n, globals()[n])
		yield
	finally:
		for n, v in saved.items():
			setattr(_core, n, v)


def process_single_video(*args, **kwargs):
	with _override_core_dependencies():
		return _core.process_single_video(*args, **kwargs)


def process_footage(*args, **kwargs):
	with _override_core_dependencies():
		return _core.process_footage(*args, **kwargs)
