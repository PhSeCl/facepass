class FacePassError(Exception):
    """Base exception for expected FacePass failures."""


class InvalidImageError(FacePassError, ValueError):
    """Raised when image bytes or arrays are invalid and should not be retried."""


class EmptyGalleryError(FacePassError):
    """Raised when no valid registration embeddings are available."""


class FatalStartupError(FacePassError):
    """Raised when the service cannot start in a usable state."""
