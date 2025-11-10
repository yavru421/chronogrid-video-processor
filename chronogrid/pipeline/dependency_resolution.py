"""
APT Pipeline Stage 2: Dependency Resolution (m2)

Resolves and validates all external dependencies required for processing.
Any missing dependency results in immediate fatal termination.
"""

import sys
import traceback
import subprocess
import shutil
from pathlib import Path
from typing import Dict, Any, List

# APT Fatal Mode Enforcement
import strict_mode  # This must be imported first in all pipeline stages

class DependencyResult:
    """Result of dependency resolution stage."""
    def __init__(self, ffmpeg_path: Path, python_deps: Dict[str, bool]):
        self.ffmpeg_path = ffmpeg_path
        self.python_deps = python_deps
        self.all_satisfied = all(python_deps.values()) and ffmpeg_path.exists()

class DependencyResolver:
    """APT Stage m2: Dependency resolution with fatal error propagation."""

    REQUIRED_PYTHON_PACKAGES = [
        'requests',
        'Pillow',
        'numpy',
        'opencv-python'
    ]

    OPTIONAL_PYTHON_PACKAGES = [
        'tkinterdnd2',  # For enhanced drag-and-drop
        'tkhtmlview',   # For markdown rendering
    ]

    @staticmethod
    def find_ffmpeg() -> Path:
        """
        Locate FFmpeg executable in system PATH.

        Returns:
            Path to FFmpeg executable

        Raises:
            SystemExit: If FFmpeg is not found (APT fatal error)
        """
        ffmpeg_path = shutil.which('ffmpeg')
        if ffmpeg_path is None:
            print("❌ APT FATAL ERROR: FFmpeg not found in system PATH")
            print("   Please install FFmpeg and ensure it's in your PATH")
            print("   Windows: Download from https://ffmpeg.org/download.html")
            print("   macOS: brew install ffmpeg")
            print("   Linux: sudo apt install ffmpeg")
            traceback.print_exc()
            sys.exit(1)

        path_obj = Path(ffmpeg_path)
        if not path_obj.exists():
            print(f"❌ APT FATAL ERROR: FFmpeg path exists but file not found: {path_obj}")
            traceback.print_exc()
            sys.exit(1)

        # Test FFmpeg by running version command
        try:
            result = subprocess.run(
                [str(path_obj), '-version'],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode != 0:
                print(f"❌ APT FATAL ERROR: FFmpeg test failed with return code {result.returncode}")
                traceback.print_exc()
                sys.exit(1)
        except subprocess.TimeoutExpired:
            print("❌ APT FATAL ERROR: FFmpeg test timed out")
            traceback.print_exc()
            sys.exit(1)
        except Exception as e:
            print(f"❌ APT FATAL ERROR: FFmpeg test failed: {e}")
            traceback.print_exc()
            sys.exit(1)

        return path_obj

    @staticmethod
    def check_python_dependencies() -> Dict[str, bool]:
        """
        Check if all required Python packages are installed.

        Returns:
            Dictionary mapping package names to availability status

        Raises:
            SystemExit: If required packages are missing (APT fatal error)
        """
        dependency_status = {}

        # Check required packages
        for package in DependencyResolver.REQUIRED_PYTHON_PACKAGES:
            try:
                __import__(package.replace('-', '_'))
                dependency_status[package] = True
            except ImportError:
                dependency_status[package] = False
                print(f"❌ APT FATAL ERROR: Required package '{package}' is not installed")
                print(f"   Run: pip install {package}")
                traceback.print_exc()
                sys.exit(1)

        # Check optional packages (warnings only, don't fail)
        for package in DependencyResolver.OPTIONAL_PYTHON_PACKAGES:
            try:
                __import__(package.replace('-', '_'))
                dependency_status[package] = True
                print(f"✓ Optional package '{package}' is available")
            except ImportError:
                dependency_status[package] = False
                print(f"⚠️  Optional package '{package}' is not available (reduced functionality)")

        return dependency_status

    @staticmethod
    def check_environment_variables() -> None:
        """
        Check for required environment variables.

        Raises:
            SystemExit: If required environment variables are missing (APT fatal error)
        """
        required_vars = ['LLAMA_API_KEY']

        for var in required_vars:
            if not var in sys.modules.get('os', {}).get('environ', {}):
                # Import os here to avoid circular imports
                import os
                if var not in os.environ:
                    print(f"❌ APT FATAL ERROR: Required environment variable '{var}' is not set")
                    print(f"   Set it with: export {var}='your-api-key'")
                    traceback.print_exc()
                    sys.exit(1)

def resolve_dependencies() -> DependencyResult:
    """
    APT Stage m2: Complete dependency resolution pipeline.

    Returns:
        DependencyResult with resolved dependencies

    Raises:
        SystemExit: On any dependency failure (APT fatal error)
    """
    try:
        # Find FFmpeg
        ffmpeg_path = DependencyResolver.find_ffmpeg()
        print(f"✓ FFmpeg found at: {ffmpeg_path}")

        # Check Python dependencies
        python_deps = DependencyResolver.check_python_dependencies()

        # Check environment variables
        DependencyResolver.check_environment_variables()
        print("✓ Environment variables validated")

        result = DependencyResult(ffmpeg_path, python_deps)

        if result.all_satisfied:
            print("✓ All dependencies resolved successfully")
        else:
            print("❌ APT FATAL ERROR: Dependency resolution failed")
            traceback.print_exc()
            sys.exit(1)

        return result

    except Exception as e:
        print(f"❌ APT FATAL ERROR: Dependency resolution failed: {e}")
        traceback.print_exc()
        sys.exit(1)