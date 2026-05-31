from .base import FaceModel
from .schemas import DetectedFace


def create_model(name: str = "insightface", **kwargs) -> FaceModel:
    normalized_name = name.lower()
    if normalized_name == "insightface":
        from .insightface_model import InsightFaceModel

        return InsightFaceModel(**kwargs)
    if normalized_name == "fake":
        from .fake_model import FakeFaceModel

        return FakeFaceModel()
    raise ValueError(f"Unsupported face model: {name}")


__all__ = ["DetectedFace", "FaceModel", "create_model"]
