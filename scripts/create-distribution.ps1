# Distribution script for Chronogrid Video Processor
# Creates a clean distribution package

param(
    [string]$OutputDir = "dist",
    [switch]$IncludeGit,
    [switch]$CreateZip
)

Write-Host "📦 Creating distribution package..." -ForegroundColor Green

# Create output directory
if (!(Test-Path $OutputDir)) {
    New-Item -ItemType Directory -Path $OutputDir | Out-Null
}

# Files to include in distribution
$filesToCopy = @(
    "README.md",
    "LICENSE",
    "requirements.txt",
    "setup.py",
    "MANIFEST.in",
    "llama_api_client.py",
    "process_footage.py",
    "run_llama_proxy_analysis.py",
    "process-video.ps1",
    "netlify-llama-proxy-openapi.yaml",
    "chronogrid-gui-pyqt.py"
)

# Copy main files
foreach ($file in $filesToCopy) {
    if (Test-Path $file) {
        Copy-Item $file $OutputDir
        Write-Host "✓ Copied: $file" -ForegroundColor Gray
    }
}

# Copy tests directory
if (Test-Path "tests") {
    Copy-Item "tests" $OutputDir -Recurse
    Write-Host "✓ Copied: tests/" -ForegroundColor Gray
}

# Copy .gitignore for reference
Copy-Item ".gitignore" $OutputDir
Write-Host "✓ Copied: .gitignore" -ForegroundColor Gray

# Create zip package if requested
if ($CreateZip) {
    $zipName = "chronogrid-video-processor-v1.0.0.zip"
    $zipPath = Join-Path $OutputDir $zipName

    Write-Host "📁 Creating zip package: $zipName" -ForegroundColor Yellow
    Compress-Archive -Path "$OutputDir\*" -DestinationPath $zipPath -Force
    Write-Host "✓ Created: $zipPath" -ForegroundColor Green
}

Write-Host "✅ Distribution package ready in: $OutputDir" -ForegroundColor Green
Write-Host ""
Write-Host "To install from this package:" -ForegroundColor Cyan
Write-Host "  pip install -r requirements.txt" -ForegroundColor White
Write-Host "  python setup.py install" -ForegroundColor White
Write-Host ""
Write-Host "To run:" -ForegroundColor Cyan
Write-Host "  chronogrid-gui    # GUI" -ForegroundColor White
Write-Host "  chronogrid video.mp4    # CLI" -ForegroundColor White
Write-Host "  .\process-video.ps1         # PowerShell script" -ForegroundColor White