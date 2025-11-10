"""
APT Pipeline Stage 1: Input Validation (m1)

Validates and normalizes all input parameters for the chronogrid processing pipeline.
Any validation failure results in immediate fatal termination.
"""

import sys
import traceback
from pathlib import Path
from typing import List, Dict, Any, NamedTuple

# APT Fatal Mode Enforcement
from chronogrid.strict_mode import *  # This must be imported first in all pipeline stages

class ValidationResult(NamedTuple):
    """Result of input validation stage."""
    is_valid: bool
    video_paths: List[Path]
    options: Dict[str, Any]
    error_message: str = ""

class InputValidator:
    """APT Stage m1: Input validation with fatal error propagation."""

    SUPPORTED_VIDEO_EXTENSIONS = {'.mp4', '.mov', '.m4v', '.avi', '.mkv'}
    SUPPORTED_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}

    @staticmethod
    def validate_video_paths(paths: List[str]) -> List[Path]:
        """
        Validate that all provided paths exist and are supported video/image files.

        Args:
            paths: List of file/directory paths to validate

        Returns:
            List of validated Path objects

        Raises:
            SystemExit: If any path is invalid (APT fatal error)
        """
        validated_paths = []

        for path_str in paths:
            path = Path(path_str)

            if not path.exists():
                print(f"❌ APT FATAL ERROR: Input path does not exist: {path}")
                traceback.print_exc()
                sys.exit(1)

            if path.is_file():
                if path.suffix.lower() not in InputValidator.SUPPORTED_VIDEO_EXTENSIONS | InputValidator.SUPPORTED_IMAGE_EXTENSIONS:
                    print(f"❌ APT FATAL ERROR: Unsupported file type: {path.suffix}")
                    traceback.print_exc()
                    sys.exit(1)
                validated_paths.append(path)

            elif path.is_dir():
                # Recursively find all supported files in directory
                found_files = []
                for ext in InputValidator.SUPPORTED_VIDEO_EXTENSIONS | InputValidator.SUPPORTED_IMAGE_EXTENSIONS:
                    found_files.extend(path.rglob(f'*{ext}'))

                if not found_files:
                    print(f"❌ APT FATAL ERROR: No supported files found in directory: {path}")
                    traceback.print_exc()
                    sys.exit(1)

                validated_paths.extend(found_files)
            else:
                print(f"❌ APT FATAL ERROR: Path is neither file nor directory: {path}")
                traceback.print_exc()
                sys.exit(1)

        if not validated_paths:
            print("❌ APT FATAL ERROR: No valid input files found")
            traceback.print_exc()
            sys.exit(1)

        return validated_paths

    @staticmethod
    def validate_options(options: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and normalize processing options.

        Args:
            options: Raw options dictionary

        Returns:
            Validated and normalized options

        Raises:
            SystemExit: If options are invalid (APT fatal error)
        """
        validated_options = {}

        # Validate frame_step
        frame_step = options.get('frame_step', 30)
        if not isinstance(frame_step, int) or frame_step < 1:
            print(f"❌ APT FATAL ERROR: Invalid frame_step: {frame_step}. Must be positive integer.")
            traceback.print_exc()
            sys.exit(1)
        validated_options['frame_step'] = frame_step

        # Validate grid_size
        grid_size = options.get('grid_size', 4)
        if not isinstance(grid_size, int) or grid_size < 2 or grid_size > 10:
            print(f"❌ APT FATAL ERROR: Invalid grid_size: {grid_size}. Must be integer between 2-10.")
            traceback.print_exc()
            sys.exit(1)
        validated_options['grid_size'] = grid_size

        # Validate analyze flag
        analyze = options.get('analyze', True)
        if not isinstance(analyze, bool):
            print(f"❌ APT FATAL ERROR: Invalid analyze flag: {analyze}. Must be boolean.")
            traceback.print_exc()
            sys.exit(1)
        validated_options['analyze'] = analyze

        # Validate output directory
        output_dir = options.get('output_dir', 'outputs')
        output_path = Path(output_dir)
        try:
            output_path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(f"❌ APT FATAL ERROR: Cannot create output directory {output_path}: {e}")
            traceback.print_exc()
            sys.exit(1)
        validated_options['output_dir'] = output_path

        return validated_options

def validate_inputs(paths: List[str], options: Dict[str, Any]) -> ValidationResult:
    """
    APT Stage m1: Complete input validation pipeline.

    Args:
        paths: List of input file/directory paths
        options: Processing options dictionary

    Returns:
        ValidationResult with validated inputs

    Raises:
        SystemExit: On any validation failure (APT fatal error)
    """
    try:
        # Validate paths
        validated_paths = InputValidator.validate_video_paths(paths)

        # Validate options
        validated_options = InputValidator.validate_options(options)

        return ValidationResult(
            is_valid=True,
            video_paths=validated_paths,
            options=validated_options
        )

    except Exception as e:
        print(f"❌ APT FATAL ERROR: Input validation failed: {e}")
        traceback.print_exc()
        sys.exit(1)