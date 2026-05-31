import io

import pytest
from PIL import Image

from src.common.errors import InvalidImageError
from src.common.images import safe_load_image
from src.common.retry import with_retry


def test_safe_load_image_rejects_corrupt_bytes() -> None:
    with pytest.raises(InvalidImageError):
        safe_load_image(b"not an image")


def test_safe_load_image_rejects_empty_bytes() -> None:
    with pytest.raises(InvalidImageError):
        safe_load_image(b"")


def test_safe_load_image_decodes_valid_image_bytes() -> None:
    buffer = io.BytesIO()
    Image.new("RGB", (2, 2), color=(255, 0, 0)).save(buffer, format="PNG")

    image = safe_load_image(buffer.getvalue())

    assert image.shape == (2, 2, 3)


def test_with_retry_retries_transient_exceptions() -> None:
    calls = {"count": 0}

    @with_retry(max_attempts=3, base_delay=0, exceptions=(ConnectionError,))
    def flaky() -> str:
        calls["count"] += 1
        if calls["count"] < 3:
            raise ConnectionError("temporary")
        return "ok"

    assert flaky() == "ok"
    assert calls["count"] == 3


def test_with_retry_does_not_retry_value_error() -> None:
    calls = {"count": 0}

    @with_retry(max_attempts=3, base_delay=0, exceptions=(ConnectionError,))
    def bad_input() -> None:
        calls["count"] += 1
        raise ValueError("deterministic")

    with pytest.raises(ValueError):
        bad_input()
    assert calls["count"] == 1
