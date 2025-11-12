# PowerShell install script for Chronogrid
# Installs Python dependencies and sets up environment

Write-Host "🔧 Installing Chronogrid dependencies..." -ForegroundColor Cyan

# Check for Python
$python = Get-Command python -ErrorAction SilentlyContinue
if (-not $python) {
    Write-Host "❌ Python not found. Please install Python 3.8+ and rerun this script." -ForegroundColor Red
    exit 1
}

# Create virtual environment
python -m venv .venv
Write-Host "✓ Created virtual environment (.venv)" -ForegroundColor Gray

# Activate virtual environment
$venvActivate = ".venv\Scripts\Activate.ps1"
if (Test-Path $venvActivate) {
    & $venvActivate
    Write-Host "✓ Activated virtual environment" -ForegroundColor Gray
} else {
    Write-Host "❌ Could not activate virtual environment. Please activate manually." -ForegroundColor Red
}

# Install requirements
pip install --upgrade pip
pip install -r requirements.txt
Write-Host "✓ Installed Python dependencies" -ForegroundColor Gray

Write-Host "✅ Chronogrid environment setup complete." -ForegroundColor Green
Write-Host "To run the GUI: python chronogrid-gui.py" -ForegroundColor White
Write-Host "To run the CLI: python -m chronogrid.interfaces.cli <video.mp4>" -ForegroundColor White
