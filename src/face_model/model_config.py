from __future__ import annotations

from pathlib import Path
from typing import Any

import tomli_w

from src.common.errors import ModelPathMissingError
from src.common.logging import get_logger

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11
    import tomli as tomllib


logger = get_logger(__name__)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


CONFIG_FILE = _project_root() / "config.toml"


def _normalize_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = _project_root() / candidate
    return candidate.resolve()


def _load_config_data() -> dict[str, Any]:
    if not CONFIG_FILE.exists():
        return {}

    try:
        with CONFIG_FILE.open("rb") as handle:
            data = tomllib.load(handle)
    except tomllib.TOMLDecodeError:
        logger.warning("Ignoring invalid TOML in %s", CONFIG_FILE)
        return {}

    if isinstance(data, dict):
        return data

    logger.warning("Ignoring unexpected config structure in %s", CONFIG_FILE)
    return {}


def load_persisted_path() -> Path | None:
    """Load the persisted absolute model path from the project-root config file."""

    data = _load_config_data()
    model_section = data.get("model")
    if not isinstance(model_section, dict):
        return None

    path_value = model_section.get("path")
    if not isinstance(path_value, str) or not path_value.strip():
        return None

    return _normalize_path(path_value)


def persist_path(path: Path) -> None:
    data = _load_config_data()
    model_section = data.get("model")
    if not isinstance(model_section, dict):
        model_section = {}
        data["model"] = model_section

    model_section["path"] = str(_normalize_path(path))
    CONFIG_FILE.write_text(tomli_w.dumps(data), encoding="utf-8")


def resolve_model_path(
    cli_path: str | Path | None = None,
    gui_path: str | Path | None = None,
) -> Path:
    """Resolve model path from CLI, GUI, then persisted config, without validating contents."""

    if cli_path is not None:
        return _normalize_path(cli_path)
    if gui_path is not None:
        return _normalize_path(gui_path)

    persisted_path = load_persisted_path()
    if persisted_path is not None:
        return persisted_path

    raise ModelPathMissingError(
        "Model path is missing. Provide --model-path, pass a GUI model path, or set [model].path in config.toml."
    )
