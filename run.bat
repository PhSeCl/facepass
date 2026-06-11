@echo off
REM ASCII-only on purpose: chcp/UTF-8 breaks on GBK consoles. This file is a thin
REM bootstrap; all real logic (env detection, deps, CPU/GPU, launch) lives in the
REM stdlib-only scripts\launcher.py so it can run under any interpreter found here.

REM Prefer Windows Terminal, and inside it PowerShell (pwsh > powershell) rather
REM than the legacy cmd window, when launched from Explorer. WT_SESSION is set by
REM Windows Terminal for its children, so this never loops; the re-entered run.bat
REM then runs the launcher in that shell. Set FACEPASS_NO_WT=1 to opt out.
REM Note the trailing "." on -d: %~dp0 ends with a backslash and "...\" would let wt
REM treat \" as an escaped quote and swallow the rest of the command line.
if defined WT_SESSION goto after_wt
if defined FACEPASS_NO_WT goto after_wt
where wt >nul 2>nul || goto after_wt
set "WTSHELL=cmd"
where pwsh >nul 2>nul && set "WTSHELL=pwsh"
if "%WTSHELL%"=="cmd" ( where powershell >nul 2>nul && set "WTSHELL=powershell" )
if "%WTSHELL%"=="cmd" (
    start "" wt.exe -d "%~dp0." cmd /c "%~nx0 %*"
) else (
    start "" wt.exe -d "%~dp0." %WTSHELL% -NoLogo -NoExit -Command "& '%~f0' %*"
)
exit /b
:after_wt

setlocal
cd /d "%~dp0"

set "VENV_PY=.venv\Scripts\python.exe"

REM Find any Python able to run the stdlib-only launcher. Probe by actually
REM running "-c import sys" instead of bare "where": that rejects the Microsoft
REM Store python.exe stub (exits non-zero) and any broken shim, and confirms the
REM interpreter really works.
set "BOOT="
if exist "%VENV_PY%" set "BOOT=%VENV_PY%"
if not defined BOOT (
    python -c "import sys" >nul 2>nul && set "BOOT=python"
)
if not defined BOOT (
    REM python.org installs may add only the "py" launcher to PATH.
    py -c "import sys" >nul 2>nul && set "BOOT=py"
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
    echo If you still cannot start it, please open an issue:
    echo   https://github.com/PhSeCl/facepass/issues
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
