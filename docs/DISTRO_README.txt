Chronogrid Video Processor - Distribution Instructions
====================================================

This package contains everything needed to run the Chronogrid PyQt GUI and CLI on Windows.

Quick Start (Windows):
----------------------
1. Ensure you have Python 3.8+ installed.
2. (Optional but recommended) Install FFmpeg and add it to your PATH.
3. Open a command prompt in this folder.
4. Create a virtual environment:
   python -m venv venv
5. Activate the virtual environment:
   venv\Scripts\activate
6. Install dependencies:
   pip install -r requirements.txt
7. Launch the GUI:
   run_chronogrid_gui.bat

Or launch manually:
   python chronogrid-gui-pyqt.py

Included Files:
---------------
- chronogrid-gui-pyqt.py      # Main PyQt GUI
- requirements.txt            # Python dependencies
- README.md / DISTRO_README.txt # Documentation
- setup.py, MANIFEST.in       # Packaging files
- create-distribution.ps1     # PowerShell distribution script
- run_chronogrid_gui.bat      # Windows batch launcher
- chronogrid/                 # Core package
- tests/                      # Test suite
- process-video.ps1           # Batch video processor (PowerShell)
- outputs/                    # Output folder

Notes:
------
- For AI analysis, you may need to set LLAMA_API_KEY as an environment variable.
- FFmpeg is required for frame extraction. Download from https://ffmpeg.org/download.html and add to PATH.
- For CLI usage, see README.md for details.

Support:
--------
See README.md or https://github.com/yavru421/chronogrid-video-processor for more info.
