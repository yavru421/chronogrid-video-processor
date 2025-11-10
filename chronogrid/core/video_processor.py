"""
Core Video Processor Module

Contains the core business logic for video processing operations.
Used by the APT pipeline stages.
"""

import sys
import traceback
from pathlib import Path
from typing import List, Dict, Any, Optional

# APT Fatal Mode Enforcement
import strict_mode

class VideoProcessingError(Exception):
    """Custom exception for video processing errors."""
    pass

class VideoProcessorCore:
    """Core video processing functionality."""

    def __init__(self, ffmpeg_path: Path):
        self.ffmpeg_path = ffmpeg_path

    def extract_frames(self, video_path: Path, output_dir: Path,
                      frame_step: int = 30, quality: int = 2) -> List[Path]:
        """
        Extract frames from video using FFmpeg.

        Args:
            video_path: Path to input video
            output_dir: Directory to save frames
            frame_step: Extract every Nth frame
            quality: JPEG quality (1-31, lower is better)

        Returns:
            List of extracted frame paths

        Raises:
            VideoProcessingError: On extraction failure
        """
        import subprocess

        output_dir.mkdir(parents=True, exist_ok=True)
        frame_pattern = output_dir / "frame_%04d.jpg"

        cmd = [
            str(self.ffmpeg_path),
            '-i', str(video_path),
            '-vf', f'select=not(mod(n\\,{frame_step}))',
            '-vsync', 'vfr',
            '-q:v', str(quality),
            str(frame_pattern)
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300
            )

            if result.returncode != 0:
                raise VideoProcessingError(f"FFmpeg failed: {result.stderr}")

            frames = sorted(output_dir.glob("frame_*.jpg"))
            if not frames:
                raise VideoProcessingError("No frames extracted")

            return frames

        except subprocess.TimeoutExpired:
            raise VideoProcessingError("Frame extraction timed out")
        except Exception as e:
            raise VideoProcessingError(f"Frame extraction failed: {e}")

    def get_video_info(self, video_path: Path) -> Dict[str, Any]:
        """
        Get video metadata using FFmpeg.

        Args:
            video_path: Path to video file

        Returns:
            Dictionary with video information

        Raises:
            VideoProcessingError: On metadata extraction failure
        """
        import subprocess
        import json

        cmd = [
            str(self.ffmpeg_path),
            '-i', str(video_path),
            '-f', 'ffmetadata',
            '-'
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60
            )

            # FFmpeg returns non-zero exit code for -f ffmetadata, but output is still valid
            if not result.stdout and result.stderr:
                # Try to parse stderr for duration info
                info = self._parse_ffmpeg_stderr(result.stderr)
                return info

            return {'raw_metadata': result.stdout.strip()}

        except Exception as e:
            raise VideoProcessingError(f"Video info extraction failed: {e}")

    def _parse_ffmpeg_stderr(self, stderr: str) -> Dict[str, Any]:
        """Parse FFmpeg stderr for basic video information."""
        info = {}

        for line in stderr.split('\n'):
            line = line.strip()
            if 'Duration:' in line:
                # Extract duration
                try:
                    duration_part = line.split('Duration:')[1].split(',')[0].strip()
                    info['duration'] = duration_part
                except:
                    pass
            elif 'Stream #' in line and 'Video:' in line:
                # Extract video stream info
                try:
                    resolution = line.split('Video:')[1].split(',')[2].strip()
                    info['resolution'] = resolution
                except:
                    pass

        return info

class ChronogridGenerator:
    """Core chronogrid image generation functionality."""

    def __init__(self):
        pass

    def create_chronogrid(self, frames: List[Path], output_path: Path,
                          grid_size: int = 4) -> Path:
        """
        Create chronogrid image from frame sequence.

        Args:
            frames: List of frame image paths
            output_path: Path to save chronogrid
            grid_size: Maximum grid dimension

        Returns:
            Path to created chronogrid

        Raises:
            VideoProcessingError: On chronogrid creation failure
        """
        try:
            from PIL import Image
            import math

            if not frames:
                raise VideoProcessingError("No frames provided")

            # Load all frames
            images = []
            for frame_path in frames:
                try:
                    img = Image.open(frame_path)
                    images.append(img)
                except Exception as e:
                    print(f"⚠️  Skipping corrupted frame {frame_path}: {e}")
                    continue

            if not images:
                raise VideoProcessingError("No valid frames found")

            # Calculate grid layout
            num_images = len(images)
            cols = min(grid_size, num_images)
            rows = math.ceil(num_images / cols)

            # Use first image as reference
            ref_img = images[0]
            img_width, img_height = ref_img.size

            # Create canvas
            grid_width = cols * img_width
            grid_height = rows * img_height
            chronogrid = Image.new('RGB', (grid_width, grid_height), (0, 0, 0))

            # Place images in grid
            for idx, img in enumerate(images):
                if idx >= cols * rows:
                    break

                row = idx // cols
                col = idx % cols

                # Resize if necessary
                if img.size != (img_width, img_height):
                    img = img.resize((img_width, img_height), Image.Resampling.LANCZOS)

                x = col * img_width
                y = row * img_height
                chronogrid.paste(img, (x, y))

            # Save chronogrid
            output_path.parent.mkdir(parents=True, exist_ok=True)
            chronogrid.save(output_path, 'JPEG', quality=95)

            return output_path

        except ImportError:
            raise VideoProcessingError("PIL (Pillow) not installed")
        except Exception as e:
            raise VideoProcessingError(f"Chronogrid creation failed: {e}")

class FileManager:
    """Core file management functionality."""

    @staticmethod
    def ensure_directory(path: Path) -> Path:
        """Ensure directory exists, create if necessary."""
        path.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def cleanup_temp_files(directory: Path, pattern: str = "*") -> int:
        """Clean up temporary files in directory."""
        cleaned = 0
        for file_path in directory.glob(pattern):
            if file_path.is_file():
                try:
                    file_path.unlink()
                    cleaned += 1
                except:
                    pass
        return cleaned

    @staticmethod
    def get_directory_size(path: Path) -> int:
        """Get total size of directory in bytes."""
        total_size = 0
        for file_path in path.rglob('*'):
            if file_path.is_file():
                try:
                    total_size += file_path.stat().st_size
                except:
                    pass
        return total_size