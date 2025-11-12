"""Chronogrid processing pipeline and AI integration utilities."""
from __future__ import annotations

import base64
import concurrent.futures
import json
import logging
import multiprocessing
import os
import re
import shutil
import subprocess
import sys
import textwrap
import urllib.request
import zipfile

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image

from chronogrid.core.api_client import LlamaAPIClient
from chronogrid.core.licensing import check_ai_analysis_allowed, record_ai_analysis

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = (".mov", ".mp4", ".m4v")
DEFAULT_FRAME_STEP = 30
DEFAULT_GRID_SIZE = 4
LLAMA_MODEL = "Llama-4-Maverick-17B-128E-Instruct-FP8"


@dataclass
class ChronogridResult:
    video_path: Path
    frames: List[Path]
    chronogrid_path: Path
    analysis_text: str | None = None


@contextmanager
def _borrow_client(client: Any):
    if hasattr(client, "__enter__") and hasattr(client, "__exit__"):
        with client as active:
            yield active
    else:
        try:
            yield client
        finally:
            close = getattr(client, "close", None)
            if callable(close):
                close()


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.strip().lower())
    return slug.strip("_")


def _strip_formatting(text: str) -> str:
    return text.replace("**", "").replace("__", "").strip()


def _extract_value(block: str, *labels: str) -> str:
    for label in labels:
        pattern = re.compile(rf"{label}\s*:\s*(.+)", re.IGNORECASE)
        match = pattern.search(block)
        if match:
            return match.group(1).strip()
    return ""


def _extract_section(text: str, header: str) -> str:
    normalized = _strip_formatting(text)
    pattern = re.compile(
        rf"(?:^|\n)\s*(?:\d+\.\s*)?{re.escape(header)}\s*:?(.*?)(?:\n\s*\n|$)",
        re.IGNORECASE | re.DOTALL,
    )
    matches = list(pattern.finditer(normalized))
    if not matches:
        return ""
    return matches[-1].group(1).strip()


def _extract_bullets(text: str, header: str, bullet_prefixes: str = "*-•") -> List[str]:
    section = _extract_section(text, header)
    items: List[str] = []
    for line in section.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if re.match(r"^\d+\.\s*", stripped):
            items.append(re.sub(r"^\d+\.\s*", "", stripped).strip())
        elif stripped[0] in bullet_prefixes:
            items.append(stripped.lstrip("*-• \t").strip())
    return items


def check_ffmpeg() -> bool:
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def get_ffmpeg_path() -> str:
    """Get the path to ffmpeg executable, checking downloaded version first."""
    # First check if ffmpeg is in PATH
    if check_ffmpeg():
        return "ffmpeg"

    # Check for downloaded version
    user_data_dir = Path.home() / ".chronogrid" / "ffmpeg"
    ffmpeg_bin = user_data_dir / "ffmpeg"
    if ffmpeg_bin.exists():
        return str(ffmpeg_bin)

    # Fallback to system PATH (should not reach here if ensure_ffmpeg was called)
    return "ffmpeg"


def get_ffprobe_path() -> str:
    """Get the path to ffprobe executable, checking downloaded version first."""
    # First check if ffprobe is in PATH
    try:
        subprocess.run(
            ["ffprobe", "-version"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return "ffprobe"
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass

    # Check for downloaded version
    user_data_dir = Path.home() / ".chronogrid" / "ffmpeg"
    ffprobe_bin = user_data_dir / "ffprobe"
    if ffprobe_bin.exists():
        return str(ffprobe_bin)

    # Fallback to system PATH
    return "ffprobe"


def download_ffmpeg() -> bool:
    """Download and extract FFmpeg binaries if not available."""
    try:
        # Determine platform
        import platform
        system = platform.system().lower()
        machine = platform.machine().lower()

        if system == "windows":
            if "64" in machine or "amd64" in machine:
                ffmpeg_url = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
                ffmpeg_dir = "ffmpeg-7.0.1-essentials_build"
            else:
                # 32-bit Windows - use older version that supports 32-bit
                ffmpeg_url = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
                ffmpeg_dir = "ffmpeg-7.0.1-essentials_build"
        elif system == "linux":
            if "64" in machine or "x86_64" in machine:
                ffmpeg_url = "https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz"
                ffmpeg_dir = "ffmpeg-7.0.1-amd64-static"
            else:
                ffmpeg_url = "https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-i686-static.tar.xz"
                ffmpeg_dir = "ffmpeg-7.0.1-i686-static"
        elif system == "darwin":  # macOS
            if "arm64" in machine or "aarch64" in machine:
                ffmpeg_url = "https://evermeet.cx/ffmpeg/ffmpeg-7.0.1.zip"
                ffmpeg_dir = "ffmpeg"
            else:
                ffmpeg_url = "https://evermeet.cx/ffmpeg/ffmpeg-7.0.1.zip"
                ffmpeg_dir = "ffmpeg"
        else:
            logger.error(f"Unsupported platform: {system} {machine}")
            return False

        # Create ffmpeg directory in user data
        user_data_dir = Path.home() / ".chronogrid"
        ffmpeg_path = user_data_dir / "ffmpeg"
        ffmpeg_path.mkdir(parents=True, exist_ok=True)

        print("Downloading FFmpeg... This may take a few minutes.")

        # Download the archive
        archive_path = ffmpeg_path / "ffmpeg_temp.zip"
        if system == "linux":
            archive_path = ffmpeg_path / "ffmpeg_temp.tar.xz"

        try:
            with urllib.request.urlopen(ffmpeg_url) as response:
                with open(archive_path, 'wb') as f:
                    shutil.copyfileobj(response, f)
        except Exception as e:
            logger.error(f"Failed to download FFmpeg: {e}")
            return False

        # Extract the archive
        try:
            if system == "windows" or system == "darwin":
                with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                    zip_ref.extractall(ffmpeg_path)
            else:  # Linux
                import tarfile
                with tarfile.open(archive_path, 'r:xz') as tar_ref:
                    tar_ref.extractall(ffmpeg_path)

            # Move binaries to correct location
            extracted_dir = ffmpeg_path / ffmpeg_dir
            if extracted_dir.exists():
                for item in extracted_dir.iterdir():
                    if item.is_file():
                        shutil.move(str(item), str(ffmpeg_path))
                    else:
                        # For directories, move contents up
                        for subitem in item.iterdir():
                            if subitem.is_file():
                                shutil.move(str(subitem), str(ffmpeg_path))

            # Clean up
            archive_path.unlink(missing_ok=True)
            if extracted_dir.exists():
                shutil.rmtree(extracted_dir)

            # Make sure binaries are executable
            for binary in ["ffmpeg", "ffprobe"]:
                bin_path = ffmpeg_path / binary
                if bin_path.exists():
                    bin_path.chmod(0o755)

            print("FFmpeg downloaded successfully!")
            return True

        except Exception as e:
            logger.error(f"Failed to extract FFmpeg: {e}")
            archive_path.unlink(missing_ok=True)
            return False

    except Exception as e:
        logger.error(f"FFmpeg download failed: {e}")
        return False


def ensure_ffmpeg() -> None:
    if not check_ffmpeg():
        print("FFmpeg not found. Attempting to download...")
        if not download_ffmpeg():
            raise RuntimeError(
                "FFmpeg is required but could not be found or downloaded. "
                "Please install FFmpeg manually from https://ffmpeg.org/download.html"
            )
        # Verify download worked
        if not check_ffmpeg():
            raise RuntimeError(
                "FFmpeg download completed but binaries are not accessible. "
                "Please install FFmpeg manually from https://ffmpeg.org/download.html"
            )


def iter_video_files(inputs: Sequence[str]) -> Iterable[Path]:
    if not inputs:
        inputs = ["."]
    for raw in inputs:
        path = Path(raw).expanduser().resolve()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield path
        elif path.is_dir():
            for candidate in sorted(path.rglob("*")):
                if candidate.is_file() and candidate.suffix.lower() in SUPPORTED_EXTENSIONS:
                    yield candidate


def list_video_files(directory: str | Path = ".") -> List[str]:
    directory = Path(directory)
    if not directory.exists():
        return []
    files: List[str] = []
    for entry in os.listdir(directory):
        full_path = directory / entry
        if full_path.is_file() and full_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            files.append(entry)
    files.sort(key=str.lower)
    return files


def detect_hardware_acceleration() -> List[str]:
    try:
        result = subprocess.run(
            [get_ffmpeg_path(), "-hwaccels"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        accelerators = [
            line.strip()
            for line in result.stdout.splitlines()
            if line.strip() and not line.lower().startswith("hardware acceleration")
        ]
        return accelerators or ["cuda"]
    except (FileNotFoundError, subprocess.CalledProcessError):
        return ["cuda"]


def get_cpu_count() -> int:
    try:
        return max(1, multiprocessing.cpu_count() - 2)
    except NotImplementedError:
        return 1


def get_optimal_ffmpeg_args(
    video_path: str | Path,
    mode: str,
    hw_accel: Sequence[str] | None = None,
) -> List[str]:
    args = ["-hide_banner", "-loglevel", "error", "-y"]
    accel = next(iter(hw_accel), None) if hw_accel else None
    if accel:
        args.extend(["-hwaccel", accel])
    args.extend(["-i", str(video_path)])

    if mode == "frames":
        args.extend(["-q:v", "2"])
    elif mode == "audio":
        filter_name = "aresample=async=1:min_hard_comp=0.100000"
        try:
            result = subprocess.run(
                [get_ffmpeg_path(), "-hide_banner", "-filters"],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            if "loudnorm" in (result.stdout or ""):
                filter_name = "loudnorm"
        except subprocess.SubprocessError:
            pass
        args.extend(["-vn", "-af", filter_name])
    else:
        raise ValueError(f"Unknown ffmpeg mode '{mode}'")
    return args


def get_video_metadata(video_path: str | Path) -> Dict[str, Any]:
    cmd = [
        get_ffprobe_path(),
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(video_path),
    ]
    try:
        result = subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return json.loads(result.stdout or "{}")
    except (subprocess.SubprocessError, json.JSONDecodeError) as exc:
        logger.debug("Unable to read metadata for %s: %s", video_path, exc)
        return {}


_SCENE_PATTERN = re.compile(r"pts_time:(?P<pts>[0-9.]+)")


def detect_scenes(video_path: str | Path, threshold: float = 0.3) -> Tuple[int, List[str], List[float]]:
    cmd = [
        get_ffmpeg_path(),
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-filter_complex",
        f"select='gt(scene,{threshold})',metadata=print",
        "-f",
        "null",
        "-",
    ]
    timestamps: List[float] = []
    try:
        result = subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        stderr_text = ""
        try:
            if isinstance(result.stderr, str):
                stderr_text = result.stderr
            elif isinstance(result.stderr, bytes):
                stderr_text = result.stderr.decode("utf-8", "ignore")
            else:
                stderr_text = str(result.stderr or "")
        except Exception:
            stderr_text = ""
        for match in _SCENE_PATTERN.finditer(stderr_text):
            timestamps.append(float(match.group("pts")))
    except subprocess.SubprocessError as exc:
        logger.debug("Scene detection failed for %s: %s", video_path, exc)

    if not timestamps:
        metadata = get_video_metadata(video_path)
        duration = float(metadata.get("format", {}).get("duration", 0) or 0)
        if duration > 0:
            segments = min(8, max(1, int(duration // 5) or 1))
            step = duration / (segments + 1)
            timestamps = [round(step * idx, 2) for idx in range(1, segments + 1)]
    return len(timestamps), [], timestamps


def extract_frames(video_path: Path, output_dir: Path, frame_step: int, timestamps: Optional[List[float]] = None) -> List[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if timestamps is not None:
        frames = []
        for i, ts in enumerate(timestamps):
            output_file = output_dir / f"{video_path.stem}_{i+1:04d}.jpg"
            cmd = [
                get_ffmpeg_path(),
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-ss",
                str(ts),
                "-i",
                str(video_path),
                "-frames:v",
                "1",
                str(output_file),
            ]
            subprocess.run(cmd, check=True)
            frames.append(output_file)
        return frames
    else:
        pattern = output_dir / f"{video_path.stem}_%04d.jpg"
        cmd = [
            get_ffmpeg_path(),
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(video_path),
            "-vf",
            f"select='not(mod(n,{frame_step}))'",
            "-vsync",
            "vfr",
            str(pattern),
        ]
        subprocess.run(cmd, check=True)
        frames = sorted(output_dir.glob(f"{video_path.stem}_*.jpg"))
        if not frames:
            raise RuntimeError(f"No frames extracted for {video_path}")
        return frames


def list_extracted_frames(directory: str | Path, stem: str) -> List[str]:
    base = Path(directory)
    if not base.exists():
        return []
    pattern = f"{stem}_*.jpg"
    return [str(path) for path in sorted(base.glob(pattern))]


def generate_chronogrid(
    frames: Sequence[str | Path],
    output_path: str | Path,
    grid_size: int = DEFAULT_GRID_SIZE,
    timestamps: Optional[List[float]] = None,
) -> Tuple[bool, List[Dict[str, Any]]]:
    frame_paths = [Path(frame) for frame in frames]
    if not frame_paths:
        return False, []

    total_tiles = grid_size * grid_size
    selected_indices = list(range(len(frame_paths)))
    if len(frame_paths) > total_tiles:
        selected_indices = [int(i * (len(frame_paths) - 1) / (total_tiles - 1)) for i in range(total_tiles)]
        frame_paths = [frame_paths[i] for i in selected_indices]
    elif len(frame_paths) < total_tiles:
        repeats = total_tiles // len(frame_paths) + 1
        selected_indices = (selected_indices * repeats)[:total_tiles]
        frame_paths = (frame_paths * repeats)[:total_tiles]

    images: List[Image.Image] = []
    for frame in frame_paths:
        with Image.open(frame) as img:
            images.append(img.copy())

    width, height = images[0].size
    grid = Image.new("RGB", (grid_size * width, grid_size * height))
    metadata: List[Dict[str, Any]] = []
    for idx, img in enumerate(images):
        row = idx // grid_size
        col = idx % grid_size
        grid.paste(img, (col * width, row * height))
        original_idx = selected_indices[idx]
        timestamp_seconds = timestamps[original_idx] if timestamps else float(idx)
        metadata.append(
            {
                "order": idx,
                "source_frame": str(frame_paths[idx]),
                "timestamp_seconds": timestamp_seconds,
                "frame_index": idx,
            }
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(output_path, "JPEG", quality=90)
    return True, metadata


def chronogrid_to_base64(image_path: Path) -> str:
    with Path(image_path).open("rb") as fp:
        return base64.b64encode(fp.read()).decode("utf-8")


def analyze_chronogrid(image_path: Path, prompt: str) -> str:
    # Check license/freemium limits
    check_ai_analysis_allowed()

    import requests
    import io

    proxy_url = "https://llama-universal-netlify-project.netlify.app/.netlify/functions/llama-proxy?path=/chat/completions"
    headers = {}
    token = os.environ.get("LLAMA_API_KEY")
    if token:
        headers["Authorization"] = f"Bearer {token}"

    # Load image to check size
    img = Image.open(image_path)
    width, height = img.size
    img.close()

    # If large, split into quadrants
    if width > 2000 or height > 2000:
        print(f"Chronogrid too large ({width}x{height}), splitting into 4 quadrants for analysis.")
        img = Image.open(image_path)
        quadrant_width = width // 2
        quadrant_height = height // 2
        quadrants = [
            img.crop((0, 0, quadrant_width, quadrant_height)),  # top-left
            img.crop((quadrant_width, 0, width, quadrant_height)),  # top-right
            img.crop((0, quadrant_height, quadrant_width, height)),  # bottom-left
            img.crop((quadrant_width, quadrant_height, width, height)),  # bottom-right
        ]
        img.close()

        analyses = []
        debug_output = []
        for i, quad in enumerate(quadrants):
            buffer = io.BytesIO()
            quad.save(buffer, format='JPEG')
            quad_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
            buffer.close()  # Close the buffer

            payload = {
                "model": LLAMA_MODEL,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": f"You are analyzing quadrant {i+1} of a split chronogrid image. {prompt}"},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{quad_b64}"}},
                        ],
                    }
                ],
                "max_completion_tokens": 800,
                "temperature": 0.2,
            }
            try:
                resp = requests.post(proxy_url, json=payload, headers=headers, timeout=120)
                resp.raise_for_status()
                quad_data = resp.json()
                debug_line = f"[APT DEBUG] quadrant {i+1} API response: {json.dumps(quad_data, indent=2)}"
                print(debug_line)
                debug_output.append(debug_line)
                analyses.append(quad_data)
            except Exception as e:
                error_line = f"❌ APT FATAL ERROR: Llama API quadrant {i+1} call failed: {e}"
                print(error_line)
                import traceback
                tb_str = traceback.format_exc()
                print(tb_str)
                debug_output.append(error_line)
                debug_output.append(tb_str)
                analyses.append({"error": str(e)})
        # Close the quadrants
        for quad in quadrants:
            quad.close()

        # Combine analyses
        combined_text = f"Combined analysis of chronogrid (split into 4 quadrants):\n\n"
        for i, analysis in enumerate(analyses):
            content = analysis.get("completion_message", {}).get("content", {}).get("text", "")
            combined_text += f"Quadrant {i+1}:\n{content}\n\n"
        debug_combined = f"[DEBUG] Combined analysis text: {combined_text}"
        print(debug_combined)
        debug_output.append(debug_combined)

        # Include all debug output in the saved text
        full_output = "\n".join(debug_output) + "\n\n" + combined_text
        record_ai_analysis()
        return full_output

    else:
        # Single API call for smaller images
        with open(image_path, "rb") as fp:
            image_data = base64.b64encode(fp.read()).decode("utf-8")

        payload = {
            "model": LLAMA_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}},
                    ],
                }
            ],
            "max_completion_tokens": 800,
            "temperature": 0.2,
        }
        try:
            resp = requests.post(proxy_url, json=payload, headers=headers, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            analysis_text = data.get("completion_message", {}).get("content", {}).get("text", "")
            if not analysis_text:
                analysis_text = "No analysis text returned from API"
        except Exception as e:
            error_msg = f"❌ APT FATAL ERROR: Llama API call failed: {e}"
            print(error_msg)
            import traceback
            tb_str = traceback.format_exc()
            print(tb_str)
            analysis_text = error_msg + "\n" + tb_str

    refusal_markers = ("cannot see images", "text-based ai")
    lowered = analysis_text.lower()
    if any(marker in lowered for marker in refusal_markers):
        logger.error("Vision model refused to analyze %s", image_path)
        raise SystemExit(2)

    record_ai_analysis()
    return analysis_text


def generate_audio_spectrogram(
    video_path: str | Path,
    output_path: str | Path,
    hw_accel: Sequence[str] | None = None,
) -> bool:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        get_ffmpeg_path(),
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-lavfi",
        "showspectrumpic=s=1280x720",
        str(output_path),
    ]
    accel = next(iter(hw_accel), None) if hw_accel else None
    if accel:
        cmd[1:1] = ["-hwaccel", accel]
    try:
        subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return output_path.exists()
    except subprocess.SubprocessError as exc:
        logger.debug("Unable to generate spectrogram for %s: %s", video_path, exc)
        return False

_ANALYSIS_PROMPTS = {
    "organization": """
        Analyze this chronological grid from {video_name}.
        CONTENT ANALYSIS:
        - Describe what is happening in each quadrant.
        CLIP ORGANIZATION:
        - Note natural breakpoints suitable for splitting or tagging.
        NAMING SUGGESTIONS:
        - Provide 3 short file names informed by on-screen action.
    """,
    "segmentation": """
        Analyze this chronological grid from {video_name}.
        VISUAL ANALYSIS OF THE GRID:
        - Identify distinct scenes and the actions captured.
        CONTENT FLOW ANALYSIS:
        - Explain how the story progresses through the frames.
        SEGMENTATION RECOMMENDATIONS:
        - Provide cut suggestions with timestamps or frame ranges.
    """,
    "metadata": """
        IMPORTANT: Extract metadata ONLY from what you can ACTUALLY SEE in {video_name}.
        VISUAL METADATA:
        - Time of day, setting clues, lighting conditions.
        ACTIVITY METADATA:
        - Subjects, actions, objects, text, signage.
        CONTENT CLASSIFICATION:
        - Genre, intended audience, usage rights signals.
    """,
    "quality": """
        IMPORTANT: Assess quality ONLY based on what you can ACTUALLY SEE in {video_name}.
        TECHNICAL QUALITY:
        - Resolution, stability, focus, exposure, noise.
        CONTENT QUALITY:
        - Composition, coverage, pacing, clarity of action.
        PRODUCTION ASSESSMENT:
        - Gear indicators, professionalism, recommended fixes.
    """,
    "summary": """
        Provide a comprehensive summary of {video_name}.
        VISUAL CONTENT ANALYSIS:
        - One-sentence description and key subjects.
        CONTENT VALUE ASSESSMENT:
        - Context, tone, and unique takeaways.
    """,
}


def get_analysis_prompt(mode: str, video_path: str | Path) -> str:
    video_name = Path(video_path).name
    template = _ANALYSIS_PROMPTS.get(mode.lower(), _ANALYSIS_PROMPTS["organization"])
    return textwrap.dedent(template).format(video_name=video_name).strip()


def parse_chronogrid_analysis(text: str) -> Dict[str, Any]:
    subject = _extract_value(text, "Primary subject matter", "Primary subject")
    activity = _extract_value(text, "Activity type", "Activity")
    setting = _extract_value(text, "Setting", "Location")
    mood = _extract_value(text, "Mood")
    tags_line = _extract_value(text, "Tags")
    tags = [tag.strip().lower() for tag in re.split(r"[,/]|\s+", tags_line) if tag.strip()]
    content_type = _extract_value(text, "Content type")
    quality = _extract_value(text, "Quality")
    suggested = _extract_value(text, "Suggested names")
    suggested_names = [_slugify(name) for name in re.split(r",|;", suggested) if name.strip()]
    split_points = _extract_bullets(text, "Split points")
    unique_features = _extract_bullets(text, "Unique features", bullet_prefixes="*•")
    return {
        "subject": subject.lower() if subject else "",
        "activity": activity.lower() if activity else "",
        "setting": setting,
        "mood": mood,
        "tags": tags,
        "content_type": content_type,
        "quality": quality,
        "suggested_names": suggested_names,
        "split_points": split_points,
        "unique_features": unique_features,
    }


def parse_segmentation_analysis(text: str) -> Dict[str, Any]:
    scenes = [item.lower() for item in _extract_bullets(text, "Visual scene analysis")]
    content_flow = _extract_section(text, "Frame-by-frame content flow")
    segments = _extract_bullets(text, "Evidence-based segmentation")
    transitions = [item.lower() for item in _extract_bullets(text, "Transitions")]
    recommended = segments.copy()
    return {
        "scenes": scenes,
        "content_flow": content_flow,
        "segments": segments,
        "recommended_cuts": recommended,
        "transitions": transitions,
    }


def parse_metadata_analysis(text: str) -> Dict[str, Any]:
    subjects = _extract_bullets(text, "Subjects")
    activities = _extract_bullets(text, "Activities")
    objects = _extract_bullets(text, "Objects")
    people = _extract_bullets(text, "People")
    setting = _extract_value(text, "Setting")
    time_of_day = _extract_value(text, "Time of day")
    lighting = _extract_value(text, "Lighting")
    genre = _extract_value(text, "Genre")
    audience = _extract_value(text, "Audience")
    return {
        "subjects": [s.lower() for s in subjects],
        "activities": [a.lower() for a in activities],
        "objects": objects,
        "people": people,
        "setting": setting,
        "time_of_day": time_of_day,
        "lighting": lighting,
        "genre": genre,
        "audience": audience,
    }


def parse_quality_analysis(text: str) -> Dict[str, Any]:
    technical_quality = _extract_value(text, "Technical quality")
    stability = _extract_value(text, "Stability")
    lighting = _extract_value(text, "Lighting")
    composition = _extract_value(text, "Composition")
    issues = _extract_bullets(text, "Issues")
    improvements = _extract_bullets(text, "Improvement suggestions")
    equipment = _extract_value(text, "Equipment")
    production_level = _extract_value(text, "Production level")
    overall = _extract_value(text, "Overall assessment")
    return {
        "technical_quality": technical_quality,
        "stability": stability,
        "lighting": lighting,
        "composition": composition,
        "issues": [item.lower() for item in issues],
        "improvements": [item.lower() for item in improvements],
        "equipment": equipment,
        "production_level": production_level,
        "overall_assessment": overall,
    }


def parse_summary_analysis(text: str) -> Dict[str, Any]:
    summary = _extract_section(text, "Visual content analysis").lower()
    context = _extract_section(text, "Visual context & setting")
    structured = _extract_section(text, "Structured visual summary")
    content_value = _extract_section(text, "Content value assessment")
    key_elements = _extract_bullets(text, "Key elements")
    timeline = "{}\n{}".format(structured, content_value).strip()
    return {
        "executive_summary": summary,
        "context": context,
        "structured_summary": structured,
        "content_value": content_value,
        "timeline": timeline,
        "key_elements": [item.lower() for item in key_elements],
    }


def parse_analysis_response(mode: str, analysis_text: str) -> Dict[str, Any]:
    mode = (mode or "organization").lower()
    if mode == "segmentation":
        return parse_segmentation_analysis(analysis_text)
    if mode == "metadata":
        return parse_metadata_analysis(analysis_text)
    if mode == "quality":
        return parse_quality_analysis(analysis_text)
    if mode == "summary":
        return parse_summary_analysis(analysis_text)
    return parse_chronogrid_analysis(analysis_text)


def _extract_from_raw_analysis(raw_text: str) -> Dict[str, Any]:
    """
    Extract meaningful information from raw AI analysis text when structured parsing fails.
    This fallback function attempts to identify subjects, activities, settings, and moods.
    """
    if not raw_text:
        return {}

    # Convert to lowercase for pattern matching
    text_lower = raw_text.lower()

    # Initialize extraction results
    extracted = {
        "subjects": [],
        "activities": [],
        "settings": [],
        "moods": [],
        "objects": [],
        "descriptions": []
    }

    # Subject patterns (people, animals, objects)
    subject_patterns = [
        r'\b(man|woman|person|people|child|children|boy|girl|adult|elderly)\b',
        r'\b(dog|cat|bird|animal|horse|car|vehicle|truck|bike)\b',
        r'\b(group|crowd|team|family|couple)\b'
    ]

    for pattern in subject_patterns:
        matches = re.findall(pattern, text_lower)
        extracted["subjects"].extend(list(set(matches)))

    # Activity patterns
    activity_patterns = [
        r'\b(walking|running|talking|working|playing|eating|drinking)\b',
        r'\b(driving|riding|traveling|moving|standing|sitting)\b',
        r'\b(working|building|creating|making|using|holding)\b'
    ]

    for pattern in activity_patterns:
        matches = re.findall(pattern, text_lower)
        extracted["activities"].extend(list(set(matches)))

    # Setting patterns
    setting_patterns = [
        r'\b(outdoor|indoor|outside|inside|street|road|park|building|house|office)\b',
        r'\b(city|urban|rural|forest|beach|mountain|desert)\b',
        r'\b(day|night|morning|evening|afternoon|sunny|cloudy|rainy)\b'
    ]

    for pattern in setting_patterns:
        matches = re.findall(pattern, text_lower)
        extracted["settings"].extend(list(set(matches)))

    # Mood/emotion patterns
    mood_patterns = [
        r'\b(happy|joyful|excited|energetic|calm|peaceful|relaxed)\b',
        r'\b(sad|angry|frustrated|worried|anxious|stressed)\b',
        r'\b(busy|intense|active|quiet|peaceful|chaotic)\b'
    ]

    for pattern in mood_patterns:
        matches = re.findall(pattern, text_lower)
        extracted["moods"].extend(list(set(matches)))

    # Object patterns
    object_patterns = [
        r'\b(phone|computer|laptop|tablet|book|paper|tool|equipment)\b',
        r'\b(food|drink|meal|snack|beverage|coffee|water)\b',
        r'\b(clothing|shirt|jacket|hat|shoes|bag|backpack)\b'
    ]

    for pattern in object_patterns:
        matches = re.findall(pattern, text_lower)
        extracted["objects"].extend(list(set(matches)))

    # Extract descriptive phrases (sentences or meaningful chunks)
    sentences = re.split(r'[.!?]+', raw_text)
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) > 10 and len(sentence) < 200:  # Reasonable length
            # Look for sentences with action words or descriptions
            if any(word in sentence.lower() for word in ['is', 'are', 'was', 'were', 'shows', 'displays', 'features', 'contains']):
                extracted["descriptions"].append(sentence)

    # Remove duplicates and empty entries
    for key in extracted:
        extracted[key] = list(set(filter(None, extracted[key])))

    return extracted


def create_semantic_summary(parsed_analysis: Dict[str, Any], mode: str, video_name: str, raw_analysis_text: str = "") -> Dict[str, Any]:
    """
    Create semantic summarization from parsed analysis data.

    Generates multiple levels of summaries with confidence scores and actionable insights.
    Falls back to raw text analysis if parsing fails.
    """
    mode = (mode or "organization").lower()

    # Base summary structure
    summary = {
        "video_name": video_name,
        "analysis_mode": mode,
        "timestamp": datetime.now().isoformat(),
        "confidence_scores": {},
        "summaries": {},
        "key_insights": [],
        "actionable_items": [],
        "semantic_clusters": {},
        "narrative_summary": ""
    }

    # Try parsed analysis first, then fall back to raw text analysis
    try:
        if mode == "organization":
            summary.update(_summarize_organization_analysis(parsed_analysis, raw_analysis_text))
        elif mode == "segmentation":
            summary.update(_summarize_segmentation_analysis(parsed_analysis, raw_analysis_text))
        elif mode == "metadata":
            summary.update(_summarize_metadata_analysis(parsed_analysis, raw_analysis_text))
        elif mode == "quality":
            summary.update(_summarize_quality_analysis(parsed_analysis, raw_analysis_text))
        elif mode == "summary":
            summary.update(_summarize_summary_analysis(parsed_analysis, raw_analysis_text))
        else:
            # Fallback for unknown modes
            summary.update(_fallback_semantic_summary(raw_analysis_text, mode))
    except Exception as e:
        logger.warning(f"Error in semantic summarization: {e}, falling back to raw analysis")
        summary.update(_fallback_semantic_summary(raw_analysis_text, mode))

    # Cross-mode semantic enhancements
    summary["semantic_clusters"] = _create_semantic_clusters(parsed_analysis, mode, raw_analysis_text)
    summary["narrative_summary"] = _generate_narrative_summary(summary)

    return summary


def _summarize_organization_analysis(data: Dict[str, Any], raw_analysis_text: str = "") -> Dict[str, Any]:
    """Create semantic summary for organization analysis."""
    summaries = {}
    key_insights = []
    actionable_items = []
    confidence_scores = {}

    # Get data from parsed analysis
    subject = data.get("subject", "").strip()
    activity = data.get("activity", "").strip()
    setting = data.get("setting", "").strip()
    mood = data.get("mood", "").strip()
    tags = data.get("tags", [])
    unique_features = data.get("unique_features", [])
    suggested_names = data.get("suggested_names", [])
    split_points = data.get("split_points", [])

    # If parsed data is empty, try to extract from raw text
    if not any([subject, activity, setting, mood]) and raw_analysis_text:
        extracted = _extract_from_raw_analysis(raw_analysis_text)
        subject = extracted.get("subjects", [subject])[0] if extracted.get("subjects") else subject
        activity = extracted.get("activities", [activity])[0] if extracted.get("activities") else activity
        setting = extracted.get("settings", [setting])[0] if extracted.get("settings") else setting
        mood = extracted.get("moods", [mood])[0] if extracted.get("moods") else mood

    # Executive summary (1-2 sentences)
    if subject and activity:
        summaries["executive"] = f"This video shows {subject} engaged in {activity}"
        if setting:
            summaries["executive"] += f" in a {setting} setting"
        if mood:
            summaries["executive"] += f", with a {mood} atmosphere"
        summaries["executive"] += "."
        confidence_scores["executive"] = 0.9 if all([subject, activity, setting]) else 0.7
    elif subject:
        summaries["executive"] = f"This video features {subject}"
        if setting:
            summaries["executive"] += f" in a {setting} setting"
        if mood:
            summaries["executive"] += f", with a {mood} atmosphere"
        summaries["executive"] += "."
        confidence_scores["executive"] = 0.7
    elif raw_analysis_text and len(raw_analysis_text.strip()) > 50:
        # Fallback to first sentence of raw analysis
        first_sentence = raw_analysis_text.strip().split('.')[0] + '.'
        summaries["executive"] = first_sentence
        confidence_scores["executive"] = 0.6
    else:
        summaries["executive"] = "Video content analysis completed."
        confidence_scores["executive"] = 0.5

    # Detailed summary
    summaries["detailed"] = ""
    if subject:
        summaries["detailed"] += f"Primary content involves {subject}"
        if activity:
            summaries["detailed"] += f" performing {activity}"
        summaries["detailed"] += ". "
    elif activity:
        summaries["detailed"] += f"Primary content involves {activity}. "

    if setting:
        summaries["detailed"] += f"The setting appears to be {setting}. "
    if mood:
        summaries["detailed"] += f"The overall mood is {mood}. "
    if tags:
        summaries["detailed"] += f"Key themes include: {', '.join(tags[:3])}."

    # Clean up detailed summary
    summaries["detailed"] = summaries["detailed"].strip()
    if not summaries["detailed"]:
        summaries["detailed"] = "Analysis of video content completed."

    # Key insights
    if subject:
        key_insights.append(f"Primary subject: {subject}")
    if activity:
        key_insights.append(f"Main activity: {activity}")
    if unique_features:
        key_insights.extend([f"Unique feature: {feature}" for feature in unique_features[:2]])
    if tags:
        key_insights.append(f"Content tags: {', '.join(tags[:3])}")

    # Actionable items
    if suggested_names:
        actionable_items.append(f"Consider renaming to: {suggested_names[0]}")
    if split_points:
        actionable_items.append(f"Potential editing points: {len(split_points)} identified")
    if not actionable_items:
        actionable_items.append("Review content for editing opportunities")

    return {
        "summaries": summaries,
        "key_insights": key_insights,
        "actionable_items": actionable_items,
        "confidence_scores": confidence_scores
    }


def _summarize_segmentation_analysis(data: Dict[str, Any], raw_analysis_text: str = "") -> Dict[str, Any]:
    """Create semantic summary for segmentation analysis."""
    summaries = {}
    key_insights = []
    actionable_items = []
    confidence_scores = {}

    scenes = data.get("scenes", [])
    segments = data.get("segments", [])
    transitions = data.get("transitions", [])

    # Executive summary
    if scenes:
        summaries["executive"] = f"Video contains {len(scenes)} distinct scenes with {len(segments)} recommended segments."
        confidence_scores["executive"] = 0.8 if scenes else 0.6
    else:
        summaries["executive"] = "Video segmentation analysis completed."
        confidence_scores["executive"] = 0.5

    # Detailed summary
    summaries["detailed"] = ""
    if scenes:
        summaries["detailed"] += f"Content flows through {len(scenes)} scenes: {', '.join(scenes[:3])}{'...' if len(scenes) > 3 else ''}. "
    if segments:
        summaries["detailed"] += f"Recommended cuts at: {', '.join(segments[:2])}."

    if not summaries["detailed"]:
        summaries["detailed"] = "Analysis of video structure and segmentation completed."

    # Key insights
    if scenes:
        key_insights.append(f"Scene progression: {' → '.join(scenes[:3])}")
    if transitions:
        key_insights.append(f"Key transitions: {', '.join(transitions[:2])}")

    # Actionable items
    if segments:
        actionable_items.extend([f"Cut suggestion: {segment}" for segment in segments[:3]])
        actionable_items.append(f"Total recommended segments: {len(segments)}")
    else:
        actionable_items.append("Review video for potential editing points")

    return {
        "summaries": summaries,
        "key_insights": key_insights,
        "actionable_items": actionable_items,
        "confidence_scores": confidence_scores
    }


def _summarize_metadata_analysis(data: Dict[str, Any], raw_analysis_text: str = "") -> Dict[str, Any]:
    """Create semantic summary for metadata analysis."""
    summaries = {}
    key_insights = []
    actionable_items = []
    confidence_scores = {}

    subjects = data.get("subjects", [])
    activities = data.get("activities", [])
    setting = data.get("setting", "").strip()
    time_of_day = data.get("time_of_day", "").strip()
    genre = data.get("genre", "").strip()
    audience = data.get("audience", "").strip()
    objects = data.get("objects", [])

    # Executive summary
    primary_subject = subjects[0] if subjects else "content"
    primary_activity = activities[0] if activities else "activity"

    if subjects or activities:
        summaries["executive"] = f"Video features {primary_subject} engaged in {primary_activity}"
        if time_of_day:
            summaries["executive"] += f" during {time_of_day}"
        if setting:
            summaries["executive"] += f" in {setting}"
        summaries["executive"] += "."
        confidence_scores["executive"] = 0.85 if (subjects and activities) else 0.7
    else:
        summaries["executive"] = "Video metadata analysis completed."
        confidence_scores["executive"] = 0.5

    # Detailed summary
    summaries["detailed"] = ""
    if subjects:
        summaries["detailed"] += f"Identified subjects: {', '.join(subjects)}. "
    if activities:
        summaries["detailed"] += f"Activities: {', '.join(activities)}. "
    if setting:
        summaries["detailed"] += f"Location: {setting}. "
    if time_of_day:
        summaries["detailed"] += f"Time: {time_of_day}."

    if not summaries["detailed"]:
        summaries["detailed"] = "Metadata extraction from video content completed."

    # Key insights
    if subjects:
        key_insights.append(f"Main subjects: {', '.join(subjects[:3])}")
    if activities:
        key_insights.append(f"Primary activities: {', '.join(activities[:3])}")
    if genre:
        key_insights.append(f"Genre classification: {genre}")

    # Actionable items
    if audience:
        actionable_items.append(f"Target audience: {audience}")
    if objects:
        actionable_items.append(f"Objects to check: {', '.join(objects[:3])}")
    if not actionable_items:
        actionable_items.append("Review metadata for content categorization")

    return {
        "summaries": summaries,
        "key_insights": key_insights,
        "actionable_items": actionable_items,
        "confidence_scores": confidence_scores
    }


def _summarize_quality_analysis(data: Dict[str, Any], raw_analysis_text: str = "") -> Dict[str, Any]:
    """Create semantic summary for quality analysis."""
    summaries = {}
    key_insights = []
    actionable_items = []
    confidence_scores = {}

    technical_quality = data.get("technical_quality", "").strip()
    issues = data.get("issues", [])
    improvements = data.get("improvements", [])
    overall = data.get("overall_assessment", "").strip()
    equipment = data.get("equipment", "").strip()
    production_level = data.get("production_level", "").strip()

    # Executive summary
    if technical_quality or overall:
        summaries["executive"] = f"Video quality is {technical_quality or 'assessed'}"
        if overall:
            summaries["executive"] += f" with overall rating: {overall}"
        summaries["executive"] += "."
        confidence_scores["executive"] = 0.9 if technical_quality else 0.7
    else:
        summaries["executive"] = "Video quality analysis completed."
        confidence_scores["executive"] = 0.5

    # Detailed summary
    summaries["detailed"] = ""
    if technical_quality:
        summaries["detailed"] += f"Technical quality: {technical_quality}. "
    if issues:
        summaries["detailed"] += f"Issues identified: {len(issues)}. "
    if improvements:
        summaries["detailed"] += f"Improvement suggestions: {len(improvements)}."

    if not summaries["detailed"]:
        summaries["detailed"] = "Technical quality assessment of video content completed."

    # Key insights
    if issues:
        key_insights.extend([f"Issue: {issue}" for issue in issues[:2]])
    if equipment:
        key_insights.append(f"Equipment detected: {equipment}")
    if production_level:
        key_insights.append(f"Production level: {production_level}")

    # Actionable items
    if improvements:
        actionable_items.extend([f"Improvement: {imp}" for imp in improvements[:3]])
    if issues:
        actionable_items.append(f"Address {len(issues)} quality issues")
    else:
        actionable_items.append("Review technical quality for potential improvements")

    return {
        "summaries": summaries,
        "key_insights": key_insights,
        "actionable_items": actionable_items,
        "confidence_scores": confidence_scores
    }


def _summarize_summary_analysis(data: Dict[str, Any], raw_analysis_text: str = "") -> Dict[str, Any]:
    """Create semantic summary for summary analysis."""
    summaries = {}
    key_insights = []
    actionable_items = []
    confidence_scores = {}

    # Extract key components
    content_summary = data.get("content_summary", "").strip()
    key_elements = data.get("key_elements", [])
    content_value = data.get("content_value", "").strip()
    themes = data.get("themes", [])
    insights = data.get("insights", [])
    actions = data.get("actionable_items", [])

    # Executive summary
    if content_summary:
        summaries["executive"] = content_summary[:200] + ("..." if len(content_summary) > 200 else "")
        confidence_scores["executive"] = 0.9
    elif content_value:
        summaries["executive"] = f"Content assessment: {content_value[:150]}{'...' if len(content_value) > 150 else ''}"
        confidence_scores["executive"] = 0.8
    else:
        summaries["executive"] = "Video content analysis completed."
        confidence_scores["executive"] = 0.5

    # Detailed summary
    parts = []
    if content_summary:
        parts.append(f"Summary: {content_summary}")
    if key_elements:
        parts.append(f"Key elements identified: {len(key_elements)}")
    if themes:
        parts.append(f"Themes: {', '.join(themes[:3])}")
    if insights:
        parts.append(f"Insights: {len(insights)}")

    summaries["detailed"] = ". ".join(parts) if parts else "Detailed content analysis completed."

    # Key insights
    if key_elements:
        key_insights.extend([f"Key element: {elem}" for elem in key_elements[:3]])
    if insights:
        key_insights.extend([f"Insight: {insight}" for insight in insights[:2]])
    if themes:
        key_insights.append(f"Primary themes: {', '.join(themes[:2])}")

    # Actionable items
    if actions:
        actionable_items.extend(actions[:3])
    if content_value:
        actionable_items.append(f"Content assessment: {content_value[:100]}...")
    else:
        actionable_items.append("Review content for key insights and themes")

    return {
        "summaries": summaries,
        "key_insights": key_insights,
        "actionable_items": actionable_items,
        "confidence_scores": confidence_scores
    }

    executive_summary = data.get("executive_summary", "")
    key_elements = data.get("key_elements", [])

    # Executive summary (use the provided one)
    summaries["executive"] = executive_summary or "Video content analysis completed."
    confidence_scores["executive"] = 0.95 if executive_summary else 0.8

    # Detailed summary
    summaries["detailed"] = data.get("structured_summary", "") or data.get("context", "")

    # Key insights
    if key_elements:
        key_insights.extend([f"Key element: {elem}" for elem in key_elements[:3]])

    # Actionable items
    content_value = data.get("content_value", "")
    if content_value:
        actionable_items.append(f"Content assessment: {content_value[:100]}...")

    return {
        "summaries": summaries,
        "key_insights": key_insights,
        "actionable_items": actionable_items,
        "confidence_scores": confidence_scores
    }


def _fallback_semantic_summary(raw_analysis_text: str, mode: str) -> Dict[str, Any]:
    """Create semantic summary from raw analysis text when structured parsing fails."""
    summaries = {}
    key_insights = []
    actionable_items = []
    confidence_scores = {}

    if not raw_analysis_text:
        summaries["executive"] = f"Analysis completed for {mode} mode."
        confidence_scores["executive"] = 0.3
        summaries["detailed"] = "No analysis text available."
        return {
            "summaries": summaries,
            "key_insights": key_insights,
            "actionable_items": actionable_items,
            "confidence_scores": confidence_scores
        }

    # Extract information using the fallback function
    extracted = _extract_from_raw_analysis(raw_analysis_text)

    # Executive summary
    if extracted.get("subjects") or extracted.get("activities"):
        subjects = extracted.get("subjects", [])
        activities = extracted.get("activities", [])
        settings = extracted.get("settings", [])

        summary_parts = []
        if subjects:
            summary_parts.append(f"features {', '.join(subjects[:2])}")
        if activities:
            summary_parts.append(f"showing {', '.join(activities[:2])}")
        if settings:
            summary_parts.append(f"in {', '.join(settings[:1])} setting")

        summaries["executive"] = f"This video {' and '.join(summary_parts)}."
        confidence_scores["executive"] = 0.6
    else:
        # Use first meaningful sentence
        sentences = [s.strip() for s in raw_analysis_text.split('.') if s.strip() and len(s.strip()) > 10]
        if sentences:
            summaries["executive"] = sentences[0] + '.'
            confidence_scores["executive"] = 0.5
        else:
            summaries["executive"] = f"Video analysis completed in {mode} mode."
            confidence_scores["executive"] = 0.4

    # Detailed summary
    parts = []
    if extracted.get("subjects"):
        parts.append(f"Subjects: {', '.join(extracted['subjects'])}")
    if extracted.get("activities"):
        parts.append(f"Activities: {', '.join(extracted['activities'])}")
    if extracted.get("settings"):
        parts.append(f"Setting: {', '.join(extracted['settings'])}")
    if extracted.get("moods"):
        parts.append(f"Mood: {', '.join(extracted['moods'])}")

    summaries["detailed"] = ". ".join(parts) if parts else "Raw analysis text processed."

    # Key insights
    if extracted.get("subjects"):
        key_insights.append(f"Identified subjects: {', '.join(extracted['subjects'][:3])}")
    if extracted.get("activities"):
        key_insights.append(f"Key activities: {', '.join(extracted['activities'][:3])}")
    if extracted.get("objects"):
        key_insights.append(f"Objects present: {', '.join(extracted['objects'][:3])}")

    # Actionable items
    if extracted.get("moods"):
        actionable_items.append(f"Content mood: {', '.join(extracted['moods'])}")
    if extracted.get("descriptions"):
        actionable_items.append("Review detailed descriptions for content insights")
    else:
        actionable_items.append("Consider re-running analysis for more structured results")

    return {
        "summaries": summaries,
        "key_insights": key_insights,
        "actionable_items": actionable_items,
        "confidence_scores": confidence_scores
    }


def _create_semantic_clusters(data: Dict[str, Any], mode: str, raw_analysis_text: str = "") -> Dict[str, Any]:
    """Create semantic clusters from analysis data."""
    clusters = {
        "temporal_patterns": [],
        "content_themes": [],
        "quality_aspects": [],
        "actionable_themes": []
    }

    # Temporal patterns
    if mode == "segmentation":
        clusters["temporal_patterns"] = data.get("scenes", []) + data.get("transitions", [])
    elif "timeline" in data:
        clusters["temporal_patterns"] = [data["timeline"]]

    # Content themes
    content_fields = ["subject", "activity", "setting", "mood", "genre", "content_type"]
    for field in content_fields:
        if field in data and data[field]:
            clusters["content_themes"].append(f"{field}: {data[field]}")

    # Quality aspects
    quality_fields = ["technical_quality", "stability", "lighting", "composition", "production_level"]
    for field in quality_fields:
        if field in data and data[field]:
            clusters["quality_aspects"].append(f"{field}: {data[field]}")

    # Actionable themes
    actionable_fields = ["issues", "improvements", "split_points", "suggested_names"]
    for field in actionable_fields:
        items = data.get(field, [])
        if items:
            clusters["actionable_themes"].extend(items[:3])

    return clusters


def _generate_narrative_summary(summary: Dict[str, Any]) -> str:
    """Generate a coherent narrative summary from all components."""
    parts = []

    # Start with executive summary
    executive = summary.get("summaries", {}).get("executive", "")
    if executive:
        parts.append(executive)

    # Add key insights
    insights = summary.get("key_insights", [])
    if insights:
        parts.append(f"Key observations: {'; '.join(insights[:3])}.")

    # Add actionable items
    actions = summary.get("actionable_items", [])
    if actions:
        parts.append(f"Recommended actions: {'; '.join(actions[:2])}.")

    # Add semantic context
    clusters = summary.get("semantic_clusters", {})
    content_themes = clusters.get("content_themes", [])
    if content_themes:
        parts.append(f"Content themes: {'; '.join(content_themes[:2])}.")

    return " ".join(parts)


def save_semantic_summary(summary: Dict[str, Any], output_path: Path) -> None:
    """Save semantic summary to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def process_video(
    video_path: str | Path,
    frame_step: int = DEFAULT_FRAME_STEP,
    grid_size: int = DEFAULT_GRID_SIZE,
    prompt: str | None = None,
    cleanup: bool = True,
) -> ChronogridResult:
    """
    Process a single video file into a chronogrid with optional AI analysis and semantic summarization.

    Args:
        video_path: Path to the video file to process
        frame_step: Extract every Nth frame (default: 30)
        grid_size: Size of the chronogrid (default: 4x4 = 16 frames)
        prompt: Optional AI analysis prompt. If provided, performs AI analysis and semantic summarization
        cleanup: Whether to delete extracted frames after chronogrid generation

    Returns:
        ChronogridResult containing paths and analysis results
    """
    ensure_ffmpeg()
    video_path = Path(video_path).expanduser().resolve()

    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # Use centralized outputs directory
    outputs_dir = Path.cwd() / "outputs"
    outputs_dir.mkdir(exist_ok=True)
    video_stem = video_path.stem
    output_dir = outputs_dir / video_stem

    # Avoid reprocessing if chronogrid exists and no analysis requested
    chronogrid_path = output_dir / f"{video_path.stem}_chronogrid.jpg"
    if chronogrid_path.exists() and not prompt:
        logger.info(f"Chronogrid already exists for {video_path.name}, skipping generation.")
        return ChronogridResult(video_path, [], chronogrid_path, None)

    # Create output directory
    if output_dir.exists():
        import shutil
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get video metadata and detect scenes
    metadata = get_video_metadata(video_path)
    duration = float(metadata.get("format", {}).get("duration", 0) or 0)

    # Calculate timestamps for uniform sampling
    timestamps: list[float] | None = None
    if duration > 0:
        num_frames = grid_size * grid_size
        step = duration / (num_frames - 1) if num_frames > 1 else 0
        timestamps = []
        for i in range(num_frames):
            ts = i * step
            if i == num_frames - 1:
                ts = min(ts, duration - 1.0)
            timestamps.append(ts)

    # Extract frames
    frames = extract_frames(video_path, output_dir, frame_step, timestamps)

    # Generate chronogrid
    success, _ = generate_chronogrid([str(frame) for frame in frames], chronogrid_path, grid_size, timestamps)
    if not success:
        raise RuntimeError(f"Failed to generate chronogrid for {video_path.name}")

    # Clean up frames if requested
    if cleanup:
        for frame in frames:
            try:
                frame.unlink()
            except OSError:
                pass

    # Perform AI analysis if prompt provided
    analysis_text = None
    if prompt:
        analysis_text = analyze_chronogrid(chronogrid_path, prompt)

        # Parse the analysis
        parsed_analysis = parse_analysis_response("organization", analysis_text)

        # Create semantic summary
        video_name = video_path.name
        semantic_summary = create_semantic_summary(parsed_analysis, "organization", video_name)

        # Save semantic summary
        summary_path = output_dir / f"{video_path.stem}_chronogrid_semantic_summary.json"
        save_semantic_summary(semantic_summary, summary_path)

        # Update analysis text to include semantic summary
        analysis_text += f"\n\n--- SEMANTIC SUMMARY ---\n{semantic_summary['narrative_summary']}"

    return ChronogridResult(video_path, frames, chronogrid_path, analysis_text)


__all__ = [
    "ChronogridResult",
    "analyze_chronogrid",
    "check_ffmpeg",
    "create_semantic_summary",
    "detect_hardware_acceleration",
    "detect_scenes",
    "ensure_ffmpeg",
    "generate_audio_spectrogram",
    "generate_chronogrid",
    "get_analysis_prompt",
    "get_cpu_count",
    "get_optimal_ffmpeg_args",
    "get_video_metadata",
    "iter_video_files",
    "list_video_files",
    "parse_analysis_response",
    "parse_chronogrid_analysis",
    "parse_metadata_analysis",
    "parse_quality_analysis",
    "parse_segmentation_analysis",
    "parse_summary_analysis",
    "process_video",
    "save_semantic_summary",
]
