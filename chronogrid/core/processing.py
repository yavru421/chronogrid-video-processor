"""Chronogrid processing pipeline and AI integration utilities."""
from __future__ import annotations

import base64
import concurrent.futures
import json
import logging
import multiprocessing
import os
import re
import subprocess
import sys
import textwrap
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from PIL import Image

from chronogrid.core.api_client import LlamaAPIClient

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


def ensure_ffmpeg() -> None:
    if not check_ffmpeg():
        raise RuntimeError("ffmpeg is required but was not found in PATH")


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
            ["ffmpeg", "-hwaccels"],
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
                ["ffmpeg", "-hide_banner", "-filters"],
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
        "ffprobe",
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
        "ffmpeg",
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
        for match in _SCENE_PATTERN.finditer(result.stderr or ""):
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


def extract_frames(video_path: Path, output_dir: Path, frame_step: int) -> List[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    pattern = output_dir / f"{video_path.stem}_%04d.jpg"
    cmd = [
        "ffmpeg",
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
) -> Tuple[bool, List[Dict[str, Any]]]:
    frame_paths = [Path(frame) for frame in frames]
    if not frame_paths:
        return False, []

    total_tiles = grid_size * grid_size
    if len(frame_paths) < total_tiles:
        repeats = total_tiles // len(frame_paths) + 1
        frame_paths = (frame_paths * repeats)[:total_tiles]
    else:
        frame_paths = frame_paths[:total_tiles]

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
        metadata.append(
            {
                "order": idx,
                "source_frame": str(frame_paths[idx]),
                "timestamp_seconds": float(idx),
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
    import requests
    import io

    proxy_url = "https://llama-universal-netlify-project.netlify.app/.netlify/functions/llama-proxy?path=/chat/completions"

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
            resp = requests.post(proxy_url, json=payload, timeout=120)
            resp.raise_for_status()
            quad_data = resp.json()
            analyses.append(quad_data)
        # Close the quadrants
        for quad in quadrants:
            quad.close()

        # Combine analyses
        combined_text = f"Combined analysis of chronogrid (split into 4 quadrants):\n\n"
        for i, analysis in enumerate(analyses):
            content = analysis.get("completion_message", {}).get("content", {}).get("text", "")
            combined_text += f"Quadrant {i+1}:\n{content}\n\n"
        return combined_text
    else:
        # Original logic for smaller images
        with image_path.open("rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("utf-8")
        payload = {
            "model": LLAMA_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                    ],
                }
            ],
            "max_completion_tokens": 800,
            "temperature": 0.2,
        }
        resp = requests.post(proxy_url, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        return data.get("completion_message", {}).get("content", {}).get("text", "")


def _extract_completion_text(payload: Any) -> str:
    if isinstance(payload, dict):
        completion = payload.get("completion_message") or {}
        content = completion.get("content")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            return " ".join(part.get("text", "") for part in content if isinstance(part, dict)).strip()
        if isinstance(content, dict):
            return content.get("text", "").strip()
    return str(payload).strip()


def analyze_chronogrid_with_llama(
    video_path: str | Path,
    video_dir: str | Path,
    base_name: str,
    chronogrid_path: str | Path,
    client_factory,
    analysis_mode: str = "organization",
    prompt: str | None = None,
) -> str:
    chronogrid_path = Path(chronogrid_path)
    video_dir = Path(video_dir)
    video_dir.mkdir(parents=True, exist_ok=True)
    mode = (analysis_mode or "organization").lower()
    prompt_text = prompt or get_analysis_prompt(mode, str(video_path))

    client = client_factory()

    def _execute(active_client: Any) -> str:
        if hasattr(active_client, "create_image_message") and hasattr(active_client, "create_chat_completion"):
            message = active_client.create_image_message(chronogrid_path, prompt_text)
            payload = {"model": LLAMA_MODEL, "messages": [message]}
            raw = active_client.create_chat_completion(payload)
            return _extract_completion_text(raw)
        encoded = chronogrid_to_base64(chronogrid_path)
        response = active_client.chat.completions.create(
            model=LLAMA_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}},
                    ],
                }
            ],
            max_completion_tokens=800,
            temperature=0.2,
        )
        return response.completion_message.content.text.strip()

    with _borrow_client(client) as active:
        analysis_text = _execute(active)

    refusal_markers = ("cannot see images", "text-based ai")
    lowered = analysis_text.lower()
    if any(marker in lowered for marker in refusal_markers):
        logger.error("Vision model refused to analyze %s", chronogrid_path)
        raise SystemExit(2)

    txt_path = video_dir / f"{base_name}_chronogrid_{mode}_analysis.txt"
    json_path = video_dir / f"{base_name}_chronogrid_{mode}_analysis.json"
    txt_path.write_text(analysis_text, encoding="utf-8")
    parsed = parse_analysis_response(mode, analysis_text)
    json_path.write_text(json.dumps(parsed, indent=2), encoding="utf-8")
    return analysis_text


def generate_audio_spectrogram(
    video_path: str | Path,
    output_path: str | Path,
    hw_accel: Sequence[str] | None = None,
) -> bool:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
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


def process_single_video(
    video_file: str | Path,
    *,
    frame_step: int = DEFAULT_FRAME_STEP,
    grid_size: int = DEFAULT_GRID_SIZE,
    hw_accel: Sequence[str] | None = None,
    analysis_mode: str | None = None,
    prompt: str | None = None,
    cleanup: bool = False,
) -> bool:
    video_path = Path(video_file).expanduser().resolve()
    if not video_path.exists():
        raise FileNotFoundError(video_path)

    logger.info(f"Analyzing codecs in {video_path}...")
    get_video_metadata(video_path)
    detect_scenes(video_path)

    output_dir = video_path.parent / video_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_pattern = output_dir / f"{video_path.stem}_%04d.jpg"
    frame_cmd = ["ffmpeg"] + get_optimal_ffmpeg_args(video_path, "frames", hw_accel)
    frame_cmd.extend([
        "-vf",
        f"select='not(mod(n,{frame_step}))'",
        "-vsync",
        "vfr",
        str(frame_pattern),
    ])
    try:
        subprocess.run(
            frame_cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except Exception as exc:
        raise RuntimeError("Frame extraction failed") from exc

    frames = list_extracted_frames(output_dir, video_path.stem)
    if not frames:
        raise RuntimeError("Frame extraction failed")

    chronogrid_path = output_dir / f"{video_path.stem}_chronogrid.jpg"
    generate_chronogrid(frames, chronogrid_path, grid_size)

    spectrogram_path = output_dir / f"{video_path.stem}_spectrogram.jpg"
    generate_audio_spectrogram(video_path, spectrogram_path, hw_accel)

    if prompt or analysis_mode:
        mode = (analysis_mode or "organization").lower()
        prompt_text = prompt or get_analysis_prompt(mode, video_path)
        analysis = analyze_chronogrid(chronogrid_path, prompt_text)
        parsed = parse_analysis_response(mode, analysis)
        (output_dir / f"{video_path.stem}_chronogrid_{mode}_analysis.json").write_text(
            json.dumps(parsed, indent=2), encoding="utf-8"
        )

    if cleanup:
        for frame in Path(output_dir).glob(f"{video_path.stem}_*.jpg"):
            if frame.name.endswith("_chronogrid.jpg"):
                continue
            try:
                frame.unlink()
            except OSError:
                pass

    return True


def process_footage(
    directory: str | Path = ".",
    *,
    max_workers: int | None = None,
    frame_step: int = DEFAULT_FRAME_STEP,
    grid_size: int = DEFAULT_GRID_SIZE,
    analysis_mode: str | None = None,
    prompt: str | None = None,
    cleanup: bool = False,
) -> List[bool]:
    if not check_ffmpeg():
        raise RuntimeError("ffmpeg is required but missing")

    videos = list_video_files(directory)
    if not videos:
        logger.info("No supported video files (.mov, .mp4, .m4v) found in current directory")
        return []

    hw_accel = detect_hardware_acceleration()
    worker_target = max_workers or get_cpu_count()
    worker_count = min(worker_target, max(1, len(videos)))
    logger.info(f"Found {len(videos)} video files to process")

    results: List[bool] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
        future_map = {
            executor.submit(
                process_single_video,
                Path(directory) / video,
                frame_step=frame_step,
                grid_size=grid_size,
                hw_accel=hw_accel,
                analysis_mode=analysis_mode,
                prompt=prompt,
                cleanup=cleanup,
            ): video
            for video in videos
        }
        for future in concurrent.futures.as_completed(future_map):
            name = future_map[future]
            try:
                results.append(future.result())
            except Exception as exc:
                logger.error("Failed to process %s: %s", name, exc)
                results.append(False)
    return results


def process_video(
    video_path: str | Path,
    frame_step: int = DEFAULT_FRAME_STEP,
    grid_size: int = DEFAULT_GRID_SIZE,
    prompt: str | None = None,
    cleanup: bool = True,
) -> ChronogridResult:
    ensure_ffmpeg()
    video_path = Path(video_path).expanduser().resolve()
    # Use centralized outputs directory - ensure absolute path
    outputs_dir = Path.cwd() / "outputs"
    outputs_dir.mkdir(exist_ok=True)
    video_stem = video_path.stem
    output_dir = outputs_dir / video_stem
    output_dir.mkdir(exist_ok=True)
    frames = extract_frames(video_path, output_dir, frame_step)
    chronogrid_path = output_dir / f"{video_path.stem}_chronogrid.jpg"
    # Skip if already processed
    if chronogrid_path.exists() and not prompt:
        logger.info(f"Chronogrid already exists for {video_path.name}, skipping.")
        return ChronogridResult(video_path, [], chronogrid_path, None)
    success, _ = generate_chronogrid([str(frame) for frame in frames], chronogrid_path, grid_size)
    if not success:
        raise RuntimeError(f"Failed to generate chronogrid for {video_path.name}")

    # Clean up frames immediately after chronogrid generation
    if cleanup:
        for frame in frames:
            try:
                frame.unlink()
            except OSError:
                pass

    analysis = analyze_chronogrid(chronogrid_path, prompt) if prompt else None

    return ChronogridResult(video_path, frames, chronogrid_path, analysis)


__all__ = [
    "ChronogridResult",
    "analyze_chronogrid",
    "analyze_chronogrid_with_llama",
    "check_ffmpeg",
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
    "process_footage",
    "process_single_video",
    "process_video",
]
