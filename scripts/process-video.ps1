param(
    [string]$Directory = ".",
    [int]$FrameStep = 30,
    [int]$GridSize = 4,
    [switch]$NoAI,
    [string]$OutputDir = "outputs",
    [switch]$KeepFrames
)

$exe = "chronogrid"

$files = Get-ChildItem -Path $Directory -File -Include *.mp4,*.mov,*.m4v -Recurse
if (-not $files) {
    Write-Host "No video files found in $Directory" -ForegroundColor Yellow
    exit 0
}

$commonArgs = @("--frame-step", $FrameStep, "--grid-size", $GridSize, "--output-dir", $OutputDir)
if ($NoAI) { $commonArgs += "--no-ai" }
if ($KeepFrames) { $commonArgs += "--keep-frames" }

foreach ($f in $files) {
    Write-Host "Processing $($f.FullName)" -ForegroundColor Cyan
    & $exe @commonArgs $f.FullName
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Failed to process $($f.FullName)" -ForegroundColor Red
        exit $LASTEXITCODE
    }
}

Write-Host "Done." -ForegroundColor Green
