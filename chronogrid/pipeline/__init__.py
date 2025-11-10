"""
APT Pipeline Orchestrator

Main entry point for the Chronogrid APT pipeline execution.
Coordinates all pipeline stages with strict fatal error propagation.
"""

import sys
import traceback
from datetime import datetime
from typing import List, Dict, Any

# APT Fatal Mode Enforcement - Must be imported first
from chronogrid.strict_mode import *

# Import pipeline stages
from .input_validation import validate_inputs, ValidationResult
from .dependency_resolution import resolve_dependencies, DependencyResult
from .processing import execute_processing, ProcessingResult
from .output_organization import organize_outputs, OutputResult

class APTPipeline:
    """Algebraic Pipeline Theory orchestrator for Chronogrid processing."""

    def __init__(self):
        print("⚙️ APT Fatal Mode Active: All exceptions are terminal.")
        print("🔬 Initializing Chronogrid APT Pipeline...")

    def execute(self, input_paths: List[str], options: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the complete APT pipeline.

        Pipeline stages:
        m1: Input validation
        m2: Dependency resolution
        m3: Processing execution
        m4: Output organization

        Any stage failure results in immediate fatal termination.

        Args:
            input_paths: List of video/image file or directory paths
            options: Processing options dictionary

        Returns:
            Pipeline execution results

        Raises:
            SystemExit: On any pipeline stage failure (APT fatal error)
        """
        try:
            print("🚀 Starting APT Pipeline Execution")
            start_time = datetime.now()

            # Stage m1: Input Validation
            print("\n📋 Stage m1: Input Validation")
            validation_result = validate_inputs(input_paths, options)

            # Stage m2: Dependency Resolution
            print("\n🔧 Stage m2: Dependency Resolution")
            dependency_result = resolve_dependencies()

            # Stage m3: Processing Execution
            print("\n⚙️  Stage m3: Processing Execution")
            processing_result = execute_processing(validation_result, dependency_result)

            # Stage m4: Output Organization
            print("\n📁 Stage m4: Output Organization")
            output_result = organize_outputs(processing_result, validation_result)

            # Pipeline completion
            end_time = datetime.now()
            duration = end_time - start_time

            final_result = {
                'success': True,
                'pipeline_version': '1.0.2',
                'execution_time': str(duration),
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'stages_completed': 4,
                'input_validation': {
                    'videos_found': len(validation_result.video_paths),
                    'options_validated': True
                },
                'dependency_resolution': {
                    'ffmpeg_found': str(dependency_result.ffmpeg_path),
                    'python_deps_satisfied': dependency_result.all_satisfied
                },
                'processing': {
                    'videos_processed': len(processing_result.output_paths),
                    'analyses_completed': len(processing_result.analysis_results)
                },
                'output_organization': {
                    'output_directory': str(output_result.output_dir),
                    'total_files': output_result.metadata['total_outputs'],
                    'files_organized': output_result.organized_files
                }
            }

            print("\n✅ APT Pipeline Execution Complete")
            print(f"⏱️  Total execution time: {duration}")
            print(f"📊 Files processed: {final_result['processing']['videos_processed']}")
            print(f"📁 Output directory: {final_result['output_organization']['output_directory']}")

            return final_result

        except Exception as e:
            print(f"\n❌ APT FATAL ERROR: Pipeline execution failed: {e}")
            traceback.print_exc()
            sys.exit(1)

def run_apt_pipeline(input_paths: List[str], options: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convenience function to run the APT pipeline.

    Args:
        input_paths: List of video/image file or directory paths
        options: Processing options dictionary

    Returns:
        Pipeline execution results

    Raises:
        SystemExit: On any pipeline failure (APT fatal error)
    """
    pipeline = APTPipeline()
    return pipeline.execute(input_paths, options)

# APT Pipeline Equation:
# y4 = m4(m3(m2(m1(x1, x2))))
# where:
#   x1 = input_paths (list of file/directory paths)
#   x2 = options (processing configuration)
#   m1 = input validation
#   m2 = dependency resolution
#   m3 = processing execution
#   m4 = output organization
#   y4 = final pipeline result or ⊥ (fatal error)