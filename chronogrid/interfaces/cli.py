#!/usr/bin/env python3
"""
Chronogrid CLI Interface

Simple command-line interface for processing videos into chronogrids.
"""

import sys
import argparse
import logging
from pathlib import Path

# Import existing modules
from chronogrid.core.processing import process_video, ensure_ffmpeg, iter_video_files

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
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )

    return parser.parse_args()


def main() -> int:
    """
    Main CLI entry point.

    Returns:
        Exit code (0 for success, 1 for error)
    """
    args = parse_arguments()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        ensure_ffmpeg()
    except RuntimeError as exc:
        logger.error(str(exc))
        return 1

    # Validate inputs for security
    for input_path in args.inputs:
        path = Path(input_path).expanduser().resolve()
        if not path.exists():
            logger.error("Input path does not exist: %s", input_path)
            return 1
        if not path.is_file() and not path.is_dir():
            logger.error("Input path is not a file or directory: %s", input_path)
            return 1

    videos = list(iter_video_files(args.inputs))
    if not videos:
        logger.warning("No video files found.")
        return 0

    logger.info(f"Found {len(videos)} video file(s) to process")

    for video in videos:
        prompt = None if args.no_ai else args.prompt
        try:
            logger.info(f"Processing {video.name}...")
            result = process_video(
                video,
                frame_step=args.frame_step,
                grid_size=args.grid_size,
                prompt=prompt,
            )
        except Exception as exc:
            logger.error("Failed to process %s: %s", video.name, exc)
            continue

        logger.info("✓ Chronogrid saved to %s", result.chronogrid_path)
        if result.analysis_text:
            analysis_file = result.chronogrid_path.parent / f"{result.chronogrid_path.stem}_analysis.txt"
            analysis_file.write_text(result.analysis_text)
            logger.info("✓ AI analysis saved to %s", analysis_file)

    logger.info("Processing complete!")
    return 0
if __name__ == '__main__':
    sys.exit(main())