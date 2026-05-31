from __future__ import annotations

from pathlib import Path
from typing import Any

import tomli_w

from src.common.logging import get_logger

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11
    import tomli as tomllib


logger = get_logger(__name__)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


CONFIG_FILE = _project_root() / "config.toml"


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

    return Path(path_value).expanduser().resolve()


def persist_path(path: Path) -> None:
    data = _load_config_data()
    model_section = data.get("model")
    if not isinstance(model_section, dict):
        model_section = {}
        data["model"] = model_section

    model_section["path"] = str(path.expanduser().resolve())
    CONFIG_FILE.write_text(tomli_w.dumps(data), encoding="utf-8")
