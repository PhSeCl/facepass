class FacePassError(Exception):
    """Base exception for expected FacePass failures."""


class ModelConfigError(FacePassError):
    """Base exception for model path configuration failures."""


class ModelPathMissingError(ModelConfigError):
    """Raised when no model path is provided by any supported source."""


class ModelNotFoundError(ModelConfigError):
    """Raised when the configured model path does not exist."""


class ModelIncompleteError(ModelConfigError):
    """Raised when the configured model files are missing or empty."""


class ModelLoadError(ModelConfigError):
    """Raised when model loading or inference fails at runtime."""


class InvalidImageError(FacePassError, ValueError):
    """Raised when image bytes or arrays are invalid and should not be retried."""


class EmptyGalleryError(FacePassError):
    """Raised when no valid registration embeddings are available."""


class FatalStartupError(FacePassError):
    """Raised when the service cannot start in a usable state."""
