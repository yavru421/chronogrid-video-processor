@echo off
echo Building Chronogrid GUI Executable...
echo.

REM Clean previous builds
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist

REM Build with PyInstaller
pyinstaller --clean chronogrid-gui-new.spec

REM Check if build succeeded
if exist "dist\chronogrid-gui-new\chronogrid-gui-new.exe" (
    echo.
    echo Build successful! Executable created at: dist\chronogrid-gui-new\chronogrid-gui-new.exe
    echo.
    for /f %%A in ('dir /s /b "dist\chronogrid-gui-new" ^| find /c ".exe"') do set EXE_COUNT=%%A
    for /f %%A in ('dir /s /b "dist\chronogrid-gui-new" ^| find /c ".dll"') do set DLL_COUNT=%%A
    for /f %%A in ('dir /s /b "dist\chronogrid-gui-new" ^| find /c ".pyd"') do set PYD_COUNT=%%A
    echo Build statistics:
    echo - EXE files: %EXE_COUNT%
    echo - DLL files: %DLL_COUNT%
    echo - PYD files: %PYD_COUNT%
    echo.
    echo Total size:
    powershell "Get-ChildItem -Path 'dist\chronogrid-gui-new' -Recurse | Measure-Object -Property Length -Sum | Select-Object -ExpandProperty Sum | ForEach-Object { '{0:N2} MB' -f ($_ / 1MB) }"
) else (
    echo.
    echo Build failed! Check the output above for errors.
    exit /b 1
)

echo.
echo Build completed successfully!