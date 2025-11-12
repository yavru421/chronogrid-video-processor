Chronogrid Video Processor - Portable EXE Distribution
========================================================

This folder contains a portable Windows executable for the Chronogrid PyQt GUI:

    dist\chronogrid-gui-pyqt.exe

How to Use:
------------
1. Double-click chronogrid-gui-pyqt.exe to launch the Chronogrid GUI.
2. No Python installation or setup required.
3. All dependencies are bundled; FFmpeg must be installed and on your PATH for frame extraction.
4. Drag and drop video files or use the Select Files button.
5. Process videos and view results in the GUI.

Notes:
------
- If you see errors about missing DLLs, ensure you have the latest Windows updates and graphics drivers.
- For AI analysis, set any required environment variables (e.g., LLAMA_API_KEY) before launching if needed.
- If you need to run on a different machine, simply copy dist\chronogrid-gui-pyqt.exe (and optionally the outputs/ folder for results).

Troubleshooting:
----------------
- If the GUI does not launch, try running as administrator or check for antivirus interference.
- If video processing fails, ensure FFmpeg is installed and accessible from the command line.

Build Process:
--------------
Built using PyInstaller with the following command:

    pyinstaller --onefile --windowed --hidden-import=chronogrid.core.processing --hidden-import=chronogrid.core.api_client --hidden-import=chronogrid.interfaces.cli --hidden-import=PIL --hidden-import=PIL.Image --hidden-import=requests --hidden-import=llama_api_client chronogrid-gui-pyqt.py

For source code and updates, see: https://github.com/yavru421/chronogrid-video-processor

MIT License. All rights reserved.
