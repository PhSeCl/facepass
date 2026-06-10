@echo off
REM ASCII-only on purpose: chcp/UTF-8 breaks on GBK consoles. This file is a thin
REM bootstrap; all real logic (env detection, deps, CPU/GPU, launch) lives in the
REM stdlib-only scripts\launcher.py so it can run under any interpreter found here.
setlocal
cd /d "%~dp0"

set "VENV_PY=.venv\Scripts\python.exe"

REM Find any Python able to run the stdlib-only launcher.
set "BOOT="
if exist "%VENV_PY%" set "BOOT=%VENV_PY%"
if not defined BOOT (
    where python >nul 2>nul && set "BOOT=python"
)
if not defined BOOT (
    REM uv can fetch a standalone interpreter without touching the project env.
    where uv >nul 2>nul && set "BOOT=uv run --no-project python"
)

if not defined BOOT (
    echo [ERROR] Neither uv nor python was found. Cannot start FacePass.
    echo Please install one of:
    echo   uv      https://docs.astral.sh/uv/
    echo   Python  https://www.python.org/
    echo.
    pause
    exit /b 1
)

%BOOT% scripts\launcher.py %*
set "EXITCODE=%errorlevel%"

REM Pause on any non-zero exit so the window does not flash and vanish.
if not "%EXITCODE%"=="0" (
    echo.
    echo [ERROR] FacePass exited with code %EXITCODE%.
    echo.
    pause
)

endlocal
exit /b %EXITCODE%
