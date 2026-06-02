import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.face_model.insightface_model import InsightFaceModel, get_runtime_diagnostics


def _collect_session_providers(model: InsightFaceModel) -> dict[str, list[str]]:
    providers: dict[str, list[str]] = {}
    for name, runtime_model in getattr(model.app, "models", {}).items():
        session = getattr(runtime_model, "session", None)
        if session is not None and hasattr(session, "get_providers"):
            providers[name] = list(session.get_providers())
    return providers


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect onnxruntime provider availability for FacePass.")
    parser.add_argument("--model-path", type=Path, default=None, help="Optional buffalo_l model directory to load.")
    args = parser.parse_args()

    report: dict[str, object] = {
        "runtime": get_runtime_diagnostics(),
    }

    if args.model_path is not None:
        model = InsightFaceModel(model_path=args.model_path)
        report["loaded_model_path"] = str(args.model_path)
        report["session_providers"] = _collect_session_providers(model)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
