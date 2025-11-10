"""
APT Pipeline Stage 3: Processing Execution (m3)

Executes the core video processing and AI analysis pipeline.
Any processing failure results in immediate fatal termination.
"""

import sys
import traceback
from pathlib import Path
from typing import List, Dict, Any, NamedTuple

# APT Fatal Mode Enforcement
import strict_mode  # This must be imported first in all pipeline stages

class ProcessingResult(NamedTuple):
    """Result of processing execution stage."""
    success: bool
    output_paths: List[Path]
    analysis_results: Dict[str, Any]
    error_message: str = ""

class VideoProcessor:
    """APT Stage m3: Core processing execution with fatal error propagation."""

    def __init__(self, ffmpeg_path: Path, options: Dict[str, Any]):
        self.ffmpeg_path = ffmpeg_path
        self.options = options

    def process_video(self, video_path: Path) -> Dict[str, Any]:
        """
        Process a single video file through the complete pipeline.

        Args:
            video_path: Path to video file to process

        Returns:
            Processing results dictionary

        Raises:
            SystemExit: On any processing failure (APT fatal error)
        """
        try:
            print(f"🎬 Processing: {video_path.name}")

            # Extract frames
            frames = self._extract_frames(video_path)
            print(f"✓ Extracted {len(frames)} frames")

            # Generate chronogrid
            chronogrid_path = self._generate_chronogrid(video_path, frames)
            print(f"✓ Generated chronogrid: {chronogrid_path}")

            # AI analysis (if enabled)
            analysis_result = {}
            if self.options.get('analyze', True):
                analysis_result = self._analyze_video(video_path, chronogrid_path)
                print("✓ AI analysis completed")

            return {
                'video_path': video_path,
                'chronogrid_path': chronogrid_path,
                'frame_count': len(frames),
                'analysis': analysis_result,
                'success': True
            }

        except Exception as e:
            print(f"❌ APT FATAL ERROR: Processing failed for {video_path}: {e}")
            traceback.print_exc()
            sys.exit(1)

    def _extract_frames(self, video_path: Path) -> List[Path]:
        """
        Extract frames from video using FFmpeg.

        Args:
            video_path: Path to video file

        Returns:
            List of extracted frame paths

        Raises:
            SystemExit: On extraction failure (APT fatal error)
        """
        import subprocess
        import tempfile

        output_dir = self.options['output_dir'] / video_path.stem
        output_dir.mkdir(parents=True, exist_ok=True)

        frame_pattern = output_dir / "frame_%04d.jpg"
        frame_step = self.options.get('frame_step', 30)

        cmd = [
            str(self.ffmpeg_path),
            '-i', str(video_path),
            '-vf', f'select=not(mod(n\\,{frame_step}))',
            '-vsync', 'vfr',
            '-q:v', '2',
            str(frame_pattern)
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )

            if result.returncode != 0:
                print(f"❌ APT FATAL ERROR: FFmpeg frame extraction failed: {result.stderr}")
                traceback.print_exc()
                sys.exit(1)

            # Find all extracted frames
            frames = sorted(output_dir.glob("frame_*.jpg"))
            if not frames:
                print("❌ APT FATAL ERROR: No frames were extracted from video")
                traceback.print_exc()
                sys.exit(1)

            return frames

        except subprocess.TimeoutExpired:
            print("❌ APT FATAL ERROR: FFmpeg frame extraction timed out")
            traceback.print_exc()
            sys.exit(1)
        except Exception as e:
            print(f"❌ APT FATAL ERROR: Frame extraction failed: {e}")
            traceback.print_exc()
            sys.exit(1)

    def _generate_chronogrid(self, video_path: Path, frames: List[Path]) -> Path:
        """
        Generate chronogrid image from extracted frames.

        Args:
            video_path: Original video path
            frames: List of extracted frame paths

        Returns:
            Path to generated chronogrid image

        Raises:
            SystemExit: On chronogrid generation failure (APT fatal error)
        """
        try:
            from PIL import Image
            import math

            if not frames:
                print("❌ APT FATAL ERROR: No frames available for chronogrid generation")
                traceback.print_exc()
                sys.exit(1)

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
                print("❌ APT FATAL ERROR: No valid frames found for chronogrid generation")
                traceback.print_exc()
                sys.exit(1)

            # Calculate grid dimensions
            grid_size = self.options.get('grid_size', 4)
            num_images = len(images)

            # Create grid layout
            cols = min(grid_size, num_images)
            rows = math.ceil(num_images / cols)

            # Use first image as reference for dimensions
            ref_img = images[0]
            img_width, img_height = ref_img.size

            # Create chronogrid canvas
            grid_width = cols * img_width
            grid_height = rows * img_height

            chronogrid = Image.new('RGB', (grid_width, grid_height), (0, 0, 0))

            # Place images in grid
            for idx, img in enumerate(images):
                if idx >= cols * rows:
                    break

                row = idx // cols
                col = idx % cols

                # Resize image if necessary
                if img.size != (img_width, img_height):
                    img = img.resize((img_width, img_height), Image.Resampling.LANCZOS)

                # Paste into grid
                x = col * img_width
                y = row * img_height
                chronogrid.paste(img, (x, y))

            # Save chronogrid
            output_dir = self.options['output_dir'] / video_path.stem
            chronogrid_path = output_dir / f"{video_path.stem}_chronogrid.jpg"

            chronogrid.save(chronogrid_path, 'JPEG', quality=95)
            return chronogrid_path

        except Exception as e:
            print(f"❌ APT FATAL ERROR: Chronogrid generation failed: {e}")
            traceback.print_exc()
            sys.exit(1)

    def _analyze_video(self, video_path: Path, chronogrid_path: Path) -> Dict[str, Any]:
        """
        Perform AI analysis on video using Llama API.

        Args:
            video_path: Original video path
            chronogrid_path: Path to chronogrid image

        Returns:
            Analysis results dictionary

        Raises:
            SystemExit: On analysis failure (APT fatal error)
        """
        try:
            import requests
            import base64
            import os

            # Load and encode chronogrid image
            with open(chronogrid_path, 'rb') as f:
                image_data = base64.b64encode(f.read()).decode('utf-8')

            # Prepare API request
            api_key = os.environ.get('LLAMA_API_KEY')
            if not api_key:
                print("❌ APT FATAL ERROR: LLAMA_API_KEY environment variable not set")
                traceback.print_exc()
                sys.exit(1)

            proxy_url = "https://llama-universal-netlify-project.netlify.app/.netlify/functions/llama-proxy?path=/chat/completions"

            payload = {
                "model": "Llama-4-Maverick-17B-128E-Instruct-FP8",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"Analyze this chronogrid visualization of video '{video_path.name}'. Describe what you see in the video content, any patterns, activities, or notable events. Be detailed and specific."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_data}"
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": 1000,
                "temperature": 0.7
            }

            headers = {
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {api_key}'
            }

            # Make API request
            response = requests.post(proxy_url, json=payload, headers=headers, timeout=120)

            if response.status_code != 200:
                print(f"❌ APT FATAL ERROR: AI analysis API request failed with status {response.status_code}: {response.text}")
                traceback.print_exc()
                sys.exit(1)

            result = response.json()

            # Extract analysis text
            analysis_text = ""
            if 'choices' in result and result['choices']:
                message = result['choices'][0].get('message', {})
                if 'content' in message:
                    analysis_text = message['content']

            return {
                'analysis_text': analysis_text,
                'api_response': result,
                'success': True
            }

        except Exception as e:
            print(f"❌ APT FATAL ERROR: AI analysis failed: {e}")
            traceback.print_exc()
            sys.exit(1)

def execute_processing(validation_result: Any, dependency_result: Any) -> ProcessingResult:
    """
    APT Stage m3: Complete processing execution pipeline.

    Args:
        validation_result: Result from input validation stage
        dependency_result: Result from dependency resolution stage

    Returns:
        ProcessingResult with all processing outputs

    Raises:
        SystemExit: On any processing failure (APT fatal error)
    """
    try:
        processor = VideoProcessor(dependency_result.ffmpeg_path, validation_result.options)

        all_results = []
        for video_path in validation_result.video_paths:
            result = processor.process_video(video_path)
            all_results.append(result)

        # Collect all output paths
        output_paths = []
        analysis_results = {}

        for result in all_results:
            if result['success']:
                output_paths.append(result['chronogrid_path'])
                analysis_results[str(result['video_path'])] = result['analysis']

        return ProcessingResult(
            success=True,
            output_paths=output_paths,
            analysis_results=analysis_results
        )

    except Exception as e:
        print(f"❌ APT FATAL ERROR: Processing execution failed: {e}")
        traceback.print_exc()
        sys.exit(1)