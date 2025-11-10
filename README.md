
# Chronogrid Video Processor

A comprehensive video processing pipeline that extracts frames, generates chronogrids, analyzes content with Llama Vision API, and provides automated Blender integration for professional video editing workflows.

## Features
- Extracts frames from video files
- Generates chronological grid images (chronogrids)
- AI-powered analysis using Llama Vision API
- CLI interface for batch processing
- Blender integration for advanced editing

## Installation
Download the latest release from the [Releases](https://github.com/yavru421/chronogrid-video-processor/releases) page.

## Usage
Run the executable from the command line:

```sh
chronogrid.exe <video_file.mp4> [options]
```

Or use Python:

```sh
python -m chronogrid <video_file.mp4> [options]
```

## Versioning
- Current version: 1.0.0
- Follows semantic versioning (MAJOR.MINOR.PATCH)

## License
See LICENSE file for details.

## Contributing
Pull requests and issues are welcome!
   ```

### Install FFmpeg

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

## Usage

### CLI

Process a single video:

```bash
python chronogrid.py video.mp4
```

Process with custom options:

```bash
python chronogrid.py video.mp4 --frame-step 15 --grid-size 5
```

Skip AI analysis:

```bash
python chronogrid.py video.mp4 --no-ai
```

### GUI

Launch the graphical interface:

```bash
python chronogrid_gui.py
```

Features:

- Drag and drop videos/folders
- Real-time progress tracking
- Built-in chronogrid preview
- Dedicated AI analysis viewer
- Batch processing support

### PowerShell Script (Windows)

For easy video processing:

```powershell
.\process-video.ps1
```

### REST API

Start the API server:

```bash
python chronogrid_api.py
```

Submit a job:

```bash
curl -X POST http://localhost:5000/api/v1/jobs \
  -H "Content-Type: application/json" \
  -d '{"video_path": "video.mp4"}'
```

## Output Structure

For input `video.mp4`, creates `outputs/video/` directory with:

- `video_chronogrid.jpg`: Chronological grid image
- `video_chronogrid_analysis.txt`: AI analysis results
- `video_chronogrid_analysis.json`: Structured analysis data

## Configuration

### Environment Variables

- `LLAMA_API_KEY`: Your Llama API key (required for AI analysis)

### Analysis Prompts

Customize AI analysis prompts in the code or via command line options.

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
- Llama API key (for analysis)
- Optional: tkinterdnd2, tkhtmlview (for enhanced GUI)

## License

MIT License - see LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## Support

For issues and questions:

- Open an issue on GitHub
- Check the troubleshooting section in the wiki

