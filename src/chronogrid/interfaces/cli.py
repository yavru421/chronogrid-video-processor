#!/usr/bin/env python3
"""
Chronogrid CLI Interface

Simple command-line interface for processing videos into chronogrids.
"""

import sys
import argparse
import logging

from chronogrid.pipeline import ChronogridPipeline, ChronogridPipelineError
from chronogrid.pipeline.models import OutputResult

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("chronogrid")


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Chronogrid - Generate chronological video grids with AI analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single video file
  python -m chronogrid.interfaces.cli video.mp4

  # Process multiple files and directories
  python -m chronogrid.interfaces.cli video1.mp4 video2.mp4 ./videos/

  # Custom grid dimensions
  python -m chronogrid.interfaces.cli --grid-size 3 video.mp4

  # Skip AI analysis
  python -m chronogrid.interfaces.cli --no-ai video.mp4

  # Custom prompt
  python -m chronogrid.interfaces.cli --prompt "Describe this video..." video.mp4
        """
    )

    parser.add_argument(
        'inputs',
        nargs='*',
        help='Video files or directories to process. Defaults to current directory.'
    )

    parser.add_argument(
        '--frame-step',
        type=int,
        default=30,
        help='Extract every Nth frame (default: 30)'
    )

    parser.add_argument(
        '--grid-size',
        type=int,
        default=4,
        help='Number of rows/columns in the chronogrid (default: 4 => 16 frames)'
    )

    parser.add_argument(
        '--prompt',
        default=(
            "You are analyzing a 4x4 chronogrid of 16 frames extracted from a video. "
            "Describe ONLY what is visually present. Identify the primary subject(s), "
            "their actions, tools, animals, and notable objects. Mention the environment, "
            "lighting, and any progression or sequence that can be inferred from frame order. "
            "Avoid speculation, metaphors, or emotional interpretation—stick to concrete facts. "
            "End with a concise timeline-style bullet list summarizing the sequence in order."
        ),
        help='Prompt sent to Llama when analyzing the chronogrid'
    )

    parser.add_argument(
        '--no-ai',
        action='store_true',
        help='Skip Llama analysis and only generate images'
    )

    parser.add_argument(
        '--output-dir',
        default='outputs',
        help='Directory where chronogrids and metadata should be written (default: outputs)'
    )

    parser.add_argument(
        '--keep-frames',
        action='store_true',
        help='Keep extracted frames on disk instead of cleaning them up'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )

    return parser.parse_args()


def _summarize(output: OutputResult) -> None:
    logger.info("Processing complete – %s video(s) handled.", len(output.files))
    logger.info("Artifacts written to %s", output.base_dir)
    if output.summary_path:
        logger.info("Summary: %s", output.summary_path)

    for bundle in output.files:
        logger.info("• %s -> %s", bundle.video_path.name, bundle.chronogrid_path)
        if bundle.analysis_path:
            logger.info("  Analysis: %s", bundle.analysis_path)
        logger.debug("  Metadata: %s", bundle.metadata_path)


def main() -> int:
    """
    Main CLI entry point.

    Returns:
        Exit code (0 for success, 1 for error)
    """
    args = parse_arguments()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    pipeline = ChronogridPipeline()
    options = {
        "frame_step": args.frame_step,
        "grid_size": args.grid_size,
        "analyze": not args.no_ai,
        "prompt": None if args.no_ai else args.prompt,
        "output_dir": args.output_dir,
        "cleanup": not args.keep_frames,
    }

    try:
        result = pipeline.run(args.inputs, **options)
    except ChronogridPipelineError as exc:
        logger.error("Pipeline failed: %s", exc)
        return 1

    _summarize(result)
    return 0


if __name__ == '__main__':
    sys.exit(main())
