import functools
import time
from collections.abc import Callable
from typing import ParamSpec, TypeVar

from .logging import get_logger


P = ParamSpec("P")
R = TypeVar("R")
logger = get_logger(__name__)


def with_retry(
    max_attempts: int = 3,
    base_delay: float = 0.5,
    exceptions: tuple[type[BaseException], ...] = (IOError, ConnectionError, TimeoutError),
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Retry transient failures with exponential backoff.

    Only exceptions listed in ``exceptions`` are retried. Deterministic input
    errors such as ValueError are intentionally not included.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            last_error: BaseException | None = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:
                    last_error = exc
                    if attempt == max_attempts:
                        break
                    delay = base_delay * (2 ** (attempt - 1))
                    logger.warning(
                        "Transient error in %s, retrying %s/%s after %.2fs: %s",
                        func.__name__,
                        attempt,
                        max_attempts,
                        delay,
                        exc,
                    )
                    if delay > 0:
                        time.sleep(delay)
            assert last_error is not None
            raise last_error

        return wrapper

    return decorator
