@echo off
REM ASCII-only on purpose: chcp/UTF-8 breaks on GBK consoles (garbled text +
REM command-parse errors). Keep all output plain English ASCII.
setlocal
cd /d "%~dp0"

set "VENV_PY=.venv\Scripts\python.exe"
set "ORT_DIR=.venv\Lib\site-packages\onnxruntime"

REM Detect uv once.
set "HAS_UV="
where uv >nul 2>nul && set "HAS_UV=1"

echo ============================================
echo   FacePass launcher
echo ============================================
echo   [C] CPU runtime  (default, portable)
echo   [G] GPU runtime  (NVIDIA CUDA; sets up onnxruntime-gpu)
echo.
choice /C CG /N /M "Select runtime [C/G]: "
if errorlevel 2 goto gpu
goto cpu


REM ============================== CPU ==============================
:cpu
set "LAUNCHER="
if defined HAS_UV set "LAUNCHER=uv run python"
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
call :check_config
echo Starting FacePass [CPU]...
echo.
%LAUNCHER% scripts\run_dev.py
set "EXITCODE=%errorlevel%"
goto after_run


REM ============================== GPU ==============================
REM GPU always launches via the venv python directly, never `uv run`, because
REM `uv run` re-syncs the locked CPU onnxruntime over the GPU build.
:gpu
if not exist "%VENV_PY%" (
    if defined HAS_UV (
        echo [INFO] No .venv yet. Creating it with uv sync...
        uv sync
        if errorlevel 1 (
            echo [ERROR] uv sync failed; cannot prepare the GPU environment.
            echo.
            pause
            exit /b 1
        )
    ) else (
        echo [ERROR] GPU mode needs a virtualenv at %VENV_PY%, and uv is not installed.
        echo Install uv and run "uv sync" first, then retry.
        echo.
        pause
        exit /b 1
    )
)

REM Fast path: skip the heavy reinstall if CUDA is already exposed.
"%VENV_PY%" -c "import onnxruntime as o,sys; sys.exit(0 if 'CUDAExecutionProvider' in o.get_available_providers() else 1)" >nul 2>nul
if not errorlevel 1 goto gpu_ready

echo [INFO] Preparing GPU onnxruntime (downloads CUDA/cuDNN; may take a while)...
echo.

REM 1) Remove any CPU/GPU onnxruntime (they share one import package).
if defined HAS_UV (
    uv pip uninstall onnxruntime onnxruntime-gpu
) else (
    "%VENV_PY%" -m pip uninstall -y onnxruntime onnxruntime-gpu
)

REM 2) Clear a leftover empty onnxruntime dir (a known half-uninstall state that
REM    makes a later install a silent no-op).
if exist "%ORT_DIR%" rd /s /q "%ORT_DIR%"

REM 3) Install the GPU build; force a real reinstall in case stale metadata remains.
if defined HAS_UV (
    uv pip install --reinstall "onnxruntime-gpu[cuda,cudnn]"
) else (
    "%VENV_PY%" -m pip install --force-reinstall "onnxruntime-gpu[cuda,cudnn]"
)
if errorlevel 1 (
    echo [ERROR] Installing onnxruntime-gpu failed.
    echo.
    pause
    exit /b 1
)

REM 4) Verify CUDA is actually available before launching.
"%VENV_PY%" -c "import onnxruntime as o,sys; sys.exit(0 if 'CUDAExecutionProvider' in o.get_available_providers() else 1)"
if errorlevel 1 (
    echo [ERROR] CUDAExecutionProvider is still not available after install.
    echo Check your NVIDIA driver / GPU, then retry.
    echo.
    pause
    exit /b 1
)

:gpu_ready
echo [OK] GPU runtime ready ^(CUDAExecutionProvider available^).
echo.
call :check_config
echo Starting FacePass [GPU]...
echo.
"%VENV_PY%" scripts\run_dev.py
set "EXITCODE=%errorlevel%"
goto after_run


REM ============================== shared tail ==============================
:after_run
if not "%EXITCODE%"=="0" (
    echo.
    echo [ERROR] FacePass exited with code %EXITCODE%.
    echo If a missing dependency was listed above, install it with uv sync or pip install.
    echo Other common causes: config.toml missing [model].path, incomplete model dir.
    echo.
    pause
)
endlocal
exit /b %EXITCODE%


:check_config
if not exist "config.toml" (
    echo [WARN] config.toml not found in this folder.
    echo        The backend needs a model path to start: set [model].path in
    echo        config.toml to your buffalo_l model directory.
    echo.
)
goto :eof
