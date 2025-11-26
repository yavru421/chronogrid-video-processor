
<img width="1024" height="1024" alt="ChatGPT Image Nov 8, 2025, 08_35_25 AM" src="https://github.com/user-attachments/assets/ffa5c071-6d42-4581-805f-617487a852af" />

# Chronogrid 
Video Processor

Chronogrid turns your video into a single chronological grid image and (optionally) sends that grid to a vision model for analysis. It ships with:

- Frame extraction (FFmpeg)
- Chronogrid generation (Pillow)
- Optional AI analysis (Netlify Llama proxy)
- A simple CLI, a Tk GUI, and a minimal REST API

## Features

- Extract frames from video files (mp4, mov, m4v)
- Generate chronological grid images (chronogrids)
- Optional AI-powered analysis using a Llama Vision proxy
- CLI interface for single or batch processing
- GUI for drag-and-drop processing and preview
- Minimal REST API for automation

## Installation

Option A: Install from source

```powershell
git clone https://github.com/yavru421/chronogrid-video-processor.git
cd chronogrid-video-processor
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
pip install .
```

Option B: Use the module without installing

```powershell
python -m pip install -r requirements.txt
python -m chronogrid.interfaces.cli --help
```

Option C: Download pre-built executable (Windows only)

Download the latest `chronogrid-gui-windows.zip` from [Releases](https://github.com/yavru421/chronogrid-video-processor/releases). Extract and run `chronogrid-gui-new.exe`.

## Building from Source

To build the Windows executable locally:

```powershell
pip install pyinstaller
.\build_exe.bat
```

The executable will be created in `dist/chronogrid-gui-new/`.

## Usage

CLI (installed):

```powershell
chronogrid <video_file.mp4> [options]
```

CLI (module):

```powershell
python -m chronogrid.interfaces.cli <video_file.mp4> [options]
```

## Versioning

- Current version: 1.1.0
- Semantic versioning (MAJOR.MINOR.PATCH)
- 1.1.0 consolidates the CLI and pipeline, and removes the legacy GitHub Actions release workflow.

### Install FFmpeg

FFmpeg is required for frame extraction.

**Windows:**

- Download from [https://ffmpeg.org/download.html](https://ffmpeg.org/download.html)
- Add to PATH

**macOS:**

```bash
brew install ffmpeg
```

**Linux:**

```bash
sudo apt install ffmpeg
```


### GUI

Launch the graphical interface:

```powershell
python chronogrid-gui-pyqt.py
```

Features:

- Drag and drop video files or select via dialog
- Real-time progress bar during processing
- Chronogrid image preview after processing
- Analysis results display
- Strict fatal error reporting (APT compliant)

### One-Step Install (Windows)

Run the install script to set up Python environment and dependencies:

```powershell
./install-chronogrid.ps1
```

Then run the GUI or CLI as above.

### PowerShell Script (Windows)

Optional helper script `process-video.ps1` is included to process all videos in the current folder using default options.

### REST API

Start the API server:

```powershell
python -m chronogrid.interfaces.api
```

Submit a job:

```bash
curl -X POST http://localhost:5000/api/v1/jobs \
  -H "Content-Type: application/json" \
  -d '{"video_path": "video.mp4"}'
```

### Pipeline API

Programmatic consumers can orchestrate processing via the `ChronogridPipeline` helper which encapsulates validation, dependency resolution, processing, and output organization:

```python
from chronogrid.pipeline import ChronogridPipeline

pipeline = ChronogridPipeline()
result = pipeline.run(
    ["./videos"],
    frame_step=15,
    grid_size=5,
    output_dir="outputs",
    analyze=True,
)

for artifact in result.files:
    print(artifact.video_path, artifact.chronogrid_path, artifact.analysis_path)
```

## Output Structure

For input `video.mp4`, creates `outputs/video/` directory with:

- `video_chronogrid.jpg`: Chronological grid image
- `video_chronogrid_analysis.txt`: AI analysis results
- `video_chronogrid_analysis.json`: Structured analysis data

## Configuration

### Environment Variables

- `LLAMA_API_KEY`: Optional. Required for the direct client in `chronogrid.core.api_client` or if your proxy expects it.

### Analysis Prompts

Customize AI analysis prompts via CLI `--prompt` or edit prompt templates in `chronogrid/core/processing.py`.

## Development

### Running Tests

```bash
pytest tests/
```

### Code Quality

```bash
# Format code
black .

# Lint code
flake8 .
```

## Requirements

- Python 3.8+
- FFmpeg
- For AI analysis (Netlify proxy is used by default in core; separate direct API client exists in `chronogrid.core.api_client`).
- Optional: tkinterdnd2, tkhtmlview (for enhanced GUI)

## License

MIT License - see LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## Notes

- Blender: The README previously referenced “Blender integration” but no Blender code currently ships in this repository. If you need it, open an issue and we’ll scope a real integration using `bpy`.

## Support

For issues and questions:

- Open an issue on GitHub

# Test workflow trigger
