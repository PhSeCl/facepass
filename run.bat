@echo off
setlocal
cd /d "%~dp0"

REM 1) Pick a launcher: prefer uv, fall back to system python.
set "LAUNCHER="
where uv >nul 2>nul && set "LAUNCHER=uv run python"
if not defined LAUNCHER (
    where python >nul 2>nul && set "LAUNCHER=python"
)

if not defined LAUNCHER (
    echo [ERROR] Neither uv nor python was found. Cannot start FacePass.
    echo Please install one of:
    echo   uv      https://docs.astral.sh/uv/
    echo   Python  https://www.python.org/
    echo.
    pause
    exit /b 1
)

if "%LAUNCHER%"=="python" (
    echo [INFO] uv not found, falling back to system python.
    echo        Installing uv is recommended for a consistent env ^(uv sync^).
    echo.
)

REM 2) Missing config.toml means the backend cannot resolve a model path.
if not exist "config.toml" (
    echo [WARN] config.toml not found in this folder.
    echo        The backend needs a model path to start: set [model].path in
    echo        config.toml to your buffalo_l model directory.
    echo.
)

echo Starting FacePass...
echo.
%LAUNCHER% scripts\run_dev.py
set "EXITCODE=%errorlevel%"

REM 3) Pause on any non-zero exit so the error stays visible (no flashing window).
if not "%EXITCODE%"=="0" (
    echo.
    echo [ERROR] FacePass exited with code %EXITCODE%.
    echo If a missing dependency was listed above, install it with uv sync or pip install.
    echo Other common causes: config.toml missing [model].path, incomplete model dir.
    echo.
    pause
)

endlocal
