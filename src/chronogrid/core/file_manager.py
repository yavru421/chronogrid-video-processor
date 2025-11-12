"""
Core File Manager Module

Contains the core business logic for file management operations.
Used by the APT pipeline stages.
"""

from pathlib import Path
from typing import List, Dict, Any, Set, Optional

class FileManagementError(Exception):
    """Custom exception for file management errors."""
    pass

class FileManager:
    """Core file management functionality."""

    SUPPORTED_VIDEO_EXTENSIONS = {'.mp4', '.mov', '.m4v', '.avi', '.mkv', '.webm'}
    SUPPORTED_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}

    @staticmethod
    def discover_files(paths: List[str]) -> List[Path]:
        """
        Discover all supported video/image files from input paths.

        Args:
            paths: List of file or directory paths

        Returns:
            List of discovered file paths

        Raises:
            FileManagementError: On discovery failure
        """
        discovered_files: List[Path] = []

        for path_str in paths:
            path = Path(path_str)

            if not path.exists():
                raise FileManagementError(f"Path does not exist: {path}")

            if path.is_file():
                if FileManager._is_supported_file(path):
                    discovered_files.append(path)
                else:
                    print(f"⚠️  Skipping unsupported file: {path}")

            elif path.is_dir():
                # Recursively find supported files
                for ext in FileManager.SUPPORTED_VIDEO_EXTENSIONS | FileManager.SUPPORTED_IMAGE_EXTENSIONS:
                    pattern = f"**/*{ext}"
                    try:
                        matches = list(path.glob(pattern))
                        discovered_files.extend(matches)
                    except Exception as e:
                        print(f"⚠️  Error scanning {pattern}: {e}")

        # Remove duplicates while preserving order
        seen: Set[Path] = set()
        unique_files: List[Path] = []
        for file_path in discovered_files:
            if file_path not in seen:
                seen.add(file_path)
                unique_files.append(file_path)

        return unique_files

    @staticmethod
    def _is_supported_file(file_path: Path) -> bool:
        """Check if file has supported extension."""
        return file_path.suffix.lower() in (FileManager.SUPPORTED_VIDEO_EXTENSIONS |
                                          FileManager.SUPPORTED_IMAGE_EXTENSIONS)

    @staticmethod
    def create_output_structure(base_dir: Path, video_files: List[Path]) -> Dict[str, Path]:
        """
        Create organized output directory structure.

        Args:
            base_dir: Base output directory
            video_files: List of video files being processed

        Returns:
            Dictionary mapping video names to their output directories

        Raises:
            FileManagementError: On structure creation failure
        """
        try:
            base_dir.mkdir(parents=True, exist_ok=True)
            output_dirs: Dict[str, Path] = {}

            for video_file in video_files:
                video_name = video_file.stem
                video_output_dir = base_dir / video_name
                video_output_dir.mkdir(parents=True, exist_ok=True)
                output_dirs[video_name] = video_output_dir

            return output_dirs

        except Exception as e:
            raise FileManagementError(f"Failed to create output structure: {e}")

    @staticmethod
    def validate_outputs(output_dirs: Dict[str, Path]) -> bool:
        """
        Validate that output directories are writable.

        Args:
            output_dirs: Dictionary of output directories

        Returns:
            True if all directories are valid

        Raises:
            FileManagementError: If directories are not writable
        """
        for name, directory in output_dirs.items():
            if not directory.exists():
                raise FileManagementError(f"Output directory does not exist: {directory}")

            # Test write access
            test_file = directory / ".write_test"
            try:
                test_file.write_text("test")
                test_file.unlink()
            except Exception as e:
                raise FileManagementError(f"Output directory not writable: {directory} - {e}")

        return True

    @staticmethod
    def cleanup_temp_files(directory: Path, patterns: Optional[List[str]] = None) -> int:
        """
        Clean up temporary files matching patterns.

        Args:
            directory: Directory to clean
            patterns: File patterns to remove (default: common temp patterns)

        Returns:
            Number of files cleaned
        """
        if patterns is None:
            patterns = ["*.tmp", "*.temp", "frame_*.jpg", ".write_test"]

        cleaned_count = 0

        for pattern in patterns:
            try:
                for file_path in directory.glob(pattern):
                    if file_path.is_file():
                        file_path.unlink()
                        cleaned_count += 1
            except Exception:
                pass  # Ignore cleanup errors

        return cleaned_count

    @staticmethod
    def get_directory_stats(directory: Path) -> Dict[str, Any]:
        """
        Get statistics about directory contents.

        Args:
            directory: Directory to analyze

        Returns:
            Dictionary with directory statistics
        """
        stats: Dict[str, Any] = {
            'total_files': 0,
            'total_size': 0,
            'file_types': {},
            'subdirectories': 0
        }

        try:
            for item in directory.rglob('*'):
                if item.is_file():
                    stats['total_files'] += 1
                    stats['total_size'] += item.stat().st_size

                    ext = item.suffix.lower() or 'no_extension'
                    stats['file_types'][ext] = stats['file_types'].get(ext, 0) + 1

                elif item.is_dir():
                    stats['subdirectories'] += 1

        except Exception:
            pass  # Ignore stat errors

        return stats

class OutputArchiver:
    """Core output archiving functionality."""

    @staticmethod
    def create_archive(output_dir: Path, archive_path: Path,
                      format: str = 'zip') -> Path:
        """
        Create compressed archive of output directory.

        Args:
            output_dir: Directory to archive
            archive_path: Path for archive file
            format: Archive format ('zip' or 'tar.gz')

        Returns:
            Path to created archive

        Raises:
            FileManagementError: On archiving failure
        """
        try:
            if format == 'zip':
                import zipfile
                with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                    for file_path in output_dir.rglob('*'):
                        if file_path.is_file():
                            zf.write(file_path, file_path.relative_to(output_dir.parent))

            elif format == 'tar.gz':
                import tarfile
                with tarfile.open(archive_path, 'w:gz') as tf:
                    tf.add(output_dir, arcname=output_dir.name)

            else:
                raise FileManagementError(f"Unsupported archive format: {format}")

            return archive_path

        except Exception as e:
            raise FileManagementError(f"Archive creation failed: {e}")

    @staticmethod
    def generate_manifest(output_dir: Path, manifest_path: Path) -> Path:
        """
        Generate manifest file listing all output files.

        Args:
            output_dir: Output directory
            manifest_path: Path for manifest file

        Returns:
            Path to created manifest

        Raises:
            FileManagementError: On manifest creation failure
        """
        try:
            import json
            from datetime import datetime

            manifest_data: Dict[str, Any] = {
                'created_at': datetime.now().isoformat(),
                'output_directory': str(output_dir),
                'files': []
            }

            for file_path in sorted(output_dir.rglob('*')):
                if file_path.is_file():
                    try:
                        stat = file_path.stat()
                        file_info: Dict[str, Any] = {
                            'path': str(file_path.relative_to(output_dir)),
                            'size': stat.st_size,
                            'modified': datetime.fromtimestamp(stat.st_mtime).isoformat()
                        }
                        manifest_data['files'].append(file_info)
                    except Exception:
                        pass  # Skip files we can't stat

            with open(manifest_path, 'w', encoding='utf-8') as f:
                json.dump(manifest_data, f, indent=2, ensure_ascii=False)

            return manifest_path

        except Exception as e:
            raise FileManagementError(f"Manifest creation failed: {e}")
