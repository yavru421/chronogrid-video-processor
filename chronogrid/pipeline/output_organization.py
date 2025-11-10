"""
APT Pipeline Stage 4: Output Organization (m4)

Organizes and validates all processing outputs.
Any output organization failure results in immediate fatal termination.
"""

import sys
import traceback
import json
from pathlib import Path
from typing import Dict, Any, List

# APT Fatal Mode Enforcement
import strict_mode  # This must be imported first in all pipeline stages

class OutputResult:
    """Result of output organization stage."""
    def __init__(self, output_dir: Path, organized_files: Dict[str, List[Path]], metadata: Dict[str, Any]):
        self.output_dir = output_dir
        self.organized_files = organized_files
        self.metadata = metadata
        self.success = True

class OutputOrganizer:
    """APT Stage m4: Output organization with fatal error propagation."""

    @staticmethod
    def organize_outputs(processing_result: Any, validation_result: Any) -> OutputResult:
        """
        Organize all processing outputs into structured directories.

        Args:
            processing_result: Result from processing stage
            validation_result: Result from validation stage

        Returns:
            OutputResult with organized outputs

        Raises:
            SystemExit: On any organization failure (APT fatal error)
        """
        try:
            base_output_dir = validation_result.options['output_dir']

            organized_files = {
                'chronogrids': [],
                'analyses': [],
                'metadata': []
            }

            metadata = {
                'processing_timestamp': None,  # Will be set by caller
                'videos_processed': len(validation_result.video_paths),
                'total_outputs': 0,
                'pipeline_version': '1.0.0',
                'options': validation_result.options
            }

            # Organize outputs by video
            for video_path in validation_result.video_paths:
                video_name = video_path.stem
                video_output_dir = base_output_dir / video_name
                video_output_dir.mkdir(parents=True, exist_ok=True)

                # Move chronogrid
                chronogrid_path = video_output_dir / f"{video_name}_chronogrid.jpg"
                if chronogrid_path.exists():
                    organized_files['chronogrids'].append(chronogrid_path)
                else:
                    print(f"❌ APT FATAL ERROR: Expected chronogrid not found: {chronogrid_path}")
                    traceback.print_exc()
                    sys.exit(1)

                # Create analysis text file
                analysis_text = processing_result.analysis_results.get(str(video_path), {}).get('analysis_text', '')
                analysis_path = video_output_dir / f"{video_name}_chronogrid_analysis.txt"

                try:
                    with open(analysis_path, 'w', encoding='utf-8') as f:
                        f.write(f"AI Analysis for: {video_name}\n")
                        f.write("=" * 50 + "\n\n")
                        f.write(analysis_text)
                    organized_files['analyses'].append(analysis_path)
                except Exception as e:
                    print(f"❌ APT FATAL ERROR: Failed to write analysis file {analysis_path}: {e}")
                    traceback.print_exc()
                    sys.exit(1)

                # Create metadata JSON file
                video_metadata = {
                    'video_path': str(video_path),
                    'video_name': video_name,
                    'chronogrid_path': str(chronogrid_path),
                    'analysis_path': str(analysis_path),
                    'processing_options': validation_result.options,
                    'analysis_result': processing_result.analysis_results.get(str(video_path), {})
                }

                metadata_path = video_output_dir / f"{video_name}_metadata.json"
                try:
                    with open(metadata_path, 'w', encoding='utf-8') as f:
                        json.dump(video_metadata, f, indent=2, ensure_ascii=False)
                    organized_files['metadata'].append(metadata_path)
                except Exception as e:
                    print(f"❌ APT FATAL ERROR: Failed to write metadata file {metadata_path}: {e}")
                    traceback.print_exc()
                    sys.exit(1)

            # Create global summary
            summary_path = base_output_dir / "processing_summary.json"
            try:
                with open(summary_path, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)
                organized_files['metadata'].append(summary_path)
            except Exception as e:
                print(f"❌ APT FATAL ERROR: Failed to write summary file {summary_path}: {e}")
                traceback.print_exc()
                sys.exit(1)

            metadata['total_outputs'] = sum(len(files) for files in organized_files.values())

            print(f"✓ Organized {metadata['total_outputs']} output files")
            print(f"✓ Processing summary: {summary_path}")

            return OutputResult(base_output_dir, organized_files, metadata)

        except Exception as e:
            print(f"❌ APT FATAL ERROR: Output organization failed: {e}")
            traceback.print_exc()
            sys.exit(1)

    @staticmethod
    def validate_outputs(output_result: OutputResult) -> bool:
        """
        Validate that all expected outputs exist and are accessible.

        Args:
            output_result: Result from output organization

        Returns:
            True if all outputs are valid

        Raises:
            SystemExit: If outputs are invalid (APT fatal error)
        """
        try:
            # Check that output directory exists
            if not output_result.output_dir.exists():
                print(f"❌ APT FATAL ERROR: Output directory does not exist: {output_result.output_dir}")
                traceback.print_exc()
                sys.exit(1)

            # Check that all organized files exist
            for category, files in output_result.organized_files.items():
                for file_path in files:
                    if not file_path.exists():
                        print(f"❌ APT FATAL ERROR: Organized file does not exist: {file_path}")
                        traceback.print_exc()
                        sys.exit(1)

                    # Check file is readable
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            f.read(1)  # Just check we can read
                    except Exception as e:
                        print(f"❌ APT FATAL ERROR: Cannot read organized file {file_path}: {e}")
                        traceback.print_exc()
                        sys.exit(1)

            print("✓ All output files validated successfully")
            return True

        except Exception as e:
            print(f"❌ APT FATAL ERROR: Output validation failed: {e}")
            traceback.print_exc()
            sys.exit(1)

def organize_outputs(processing_result: Any, validation_result: Any) -> OutputResult:
    """
    APT Stage m4: Complete output organization pipeline.

    Args:
        processing_result: Result from processing stage
        validation_result: Result from validation stage

    Returns:
        OutputResult with organized and validated outputs

    Raises:
        SystemExit: On any organization failure (APT fatal error)
    """
    try:
        # Organize outputs
        output_result = OutputOrganizer.organize_outputs(processing_result, validation_result)

        # Validate outputs
        OutputOrganizer.validate_outputs(output_result)

        return output_result

    except Exception as e:
        print(f"❌ APT FATAL ERROR: Output organization failed: {e}")
        traceback.print_exc()
        sys.exit(1)