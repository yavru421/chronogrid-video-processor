Write-Host "Building Chronogrid GUI..." -ForegroundColor Green
Write-Host "Installing dependencies..." -ForegroundColor Yellow
pip install PyQt5 Pillow requests flask pyinstaller
Write-Host "Building executable..." -ForegroundColor Yellow
pyinstaller --onefile --windowed --exclude-module PyQt6 --exclude-module PyQt6.Qt6 --exclude-module PyQt6.sip src/chronogrid/interfaces/gui.py --hidden-import PyQt5.QtCore --hidden-import PyQt5.QtWidgets --hidden-import PyQt5.QtGui
Write-Host "Build complete! Check dist/ folder" -ForegroundColor Green
