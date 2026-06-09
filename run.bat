@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

REM 1) 选择启动器：优先 uv，没有 uv 则退回系统 python
set "LAUNCHER="
where uv >nul 2>nul && set "LAUNCHER=uv run python"
if not defined LAUNCHER (
    where python >nul 2>nul && set "LAUNCHER=python"
)

if not defined LAUNCHER (
    echo [错误] 未找到 uv，也未找到 python，无法启动 FacePass。
    echo 请安装其一：
    echo   uv      https://docs.astral.sh/uv/
    echo   Python  https://www.python.org/
    echo.
    pause
    exit /b 1
)

if "%LAUNCHER%"=="python" (
    echo [提示] 未找到 uv，已改用系统 python 运行。
    echo 建议安装 uv 以获得一致的依赖环境（uv sync 可自动装齐依赖）。
    echo.
)

REM 2) 缺少 config.toml 会导致后端找不到模型路径而启动失败，提前提示
if not exist "config.toml" (
    echo [警告] 当前目录未找到 config.toml，后端可能因缺少模型路径而启动失败。
    echo 请在 config.toml 中设置 [model].path 指向 buffalo_l 模型目录。
    echo.
)

echo 正在启动 FacePass，请稍候...
echo.
%LAUNCHER% scripts\run_dev.py
set "EXITCODE=%errorlevel%"

REM 3) 任何非零退出都停下来显示错误码，避免窗口闪退看不到日志
if not "%EXITCODE%"=="0" (
    echo.
    echo [错误] FacePass 异常退出，错误码 %EXITCODE%。
    echo 若上方提示缺少依赖库，请按提示用 uv sync 或 pip install 安装。
    echo 其它常见原因：config.toml 缺少 [model].path、模型目录不完整。
    echo.
    pause
)

endlocal
