#!/usr/bin/env python3
"""
Setup script for Chronogrid Video Processor
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

# Read requirements
requirements = []
if (this_directory / "requirements.txt").exists():
    requirements = (this_directory / "requirements.txt").read_text(encoding="utf-8").strip().split('\n')

setup(
    name="chronogrid-video-processor",
    version="1.1.0",
    author="Chronogrid Team",
    author_email="",
    description="Generate chronogrids from videos with AI analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yavru421/chronogrid-video-processor",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: End Users/Desktop",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Multimedia :: Video",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "gui": ["tkinterdnd2", "tkhtmlview", "Pillow"],
        "dev": ["pytest", "pytest-cov", "black", "flake8"],
    },
    entry_points={
        "console_scripts": [
            "chronogrid=chronogrid.interfaces.cli:main",
            "chronogrid-gui=chronogrid.interfaces.gui:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
