"""Dependency resolution for the Chronogrid pipeline."""

from __future__ import annotations

import importlib
import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Iterable, Sequence

from .errors import ChronogridPipelineError
from .models import DependencyResult

REQUIRED_PYTHON_PACKAGES = ("requests", "Pillow", "numpy", "opencv-python")
OPTIONAL_PYTHON_PACKAGES = ("tkinterdnd2", "tkhtmlview")
ENV_VARS = ("LLAMA_API_KEY",)
IMPORT_NAME_OVERRIDES = {
    "Pillow": "PIL",
    "opencv-python": "cv2",
}


def _import_name(package: str) -> str:
    return IMPORT_NAME_OVERRIDES.get(package, package.replace("-", "_"))


def _require_packages(packages: Sequence[str], *, stage: str) -> Dict[str, bool]:
    statuses: Dict[str, bool] = {}
    missing: list[str] = []

    for package in packages:
        module_name = _import_name(package)
        try:
            importlib.import_module(module_name)
            statuses[package] = True
        except ModuleNotFoundError:
            statuses[package] = False
            missing.append(package)

    if missing:
        deps = ", ".join(missing)
        raise ChronogridPipelineError(
            f"Missing required Python packages: {deps}. Run `pip install {deps}`",
            stage=stage,
        )

    return statuses


def _test_optional_packages(packages: Sequence[str]) -> Dict[str, bool]:
    statuses: Dict[str, bool] = {}
    for package in packages:
        module_name = _import_name(package)
        try:
            importlib.import_module(module_name)
            statuses[package] = True
        except ModuleNotFoundError:
            statuses[package] = False
    return statuses


def _verify_ffmpeg() -> Path:
    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        raise ChronogridPipelineError(
            "ffmpeg was not found in PATH. Install it from https://ffmpeg.org/ and retry.",
            stage="dependency_resolution",
        )

    path = Path(ffmpeg_path)
    try:
        subprocess.run(
            [str(path), "-version"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except subprocess.SubprocessError as exc:
        raise ChronogridPipelineError(
            f"ffmpeg exists at {path} but could not be executed: {exc}",
            stage="dependency_resolution",
        ) from exc
    return path


def _collect_env(require_analysis: bool) -> Dict[str, str]:
    env: Dict[str, str] = {}
    if not require_analysis:
        return env

    missing: list[str] = []
    for name in ENV_VARS:
        value = os.environ.get(name)
        if value:
            env[name] = value
        else:
            missing.append(name)

    if missing:
        raise ChronogridPipelineError(
            f"Missing required environment variable(s): {', '.join(missing)}",
            stage="dependency_resolution",
        )
    return env


def resolve_dependencies(require_analysis: bool = True) -> DependencyResult:
    """Ensure ffmpeg, python deps, and environment variables are present."""
    ffmpeg = _verify_ffmpeg()
    required = _require_packages(REQUIRED_PYTHON_PACKAGES, stage="dependency_resolution")
    optional = _test_optional_packages(OPTIONAL_PYTHON_PACKAGES)
    env = _collect_env(require_analysis)

    python_deps = {**required, **optional}
    return DependencyResult(ffmpeg_path=ffmpeg, python_deps=python_deps, environment=env)


__all__ = [
    "resolve_dependencies",
    "REQUIRED_PYTHON_PACKAGES",
    "OPTIONAL_PYTHON_PACKAGES",
]
