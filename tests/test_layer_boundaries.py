import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_package(package: str) -> str:
    return "\n".join(path.read_text(encoding="utf-8") for path in (ROOT / "src" / package).glob("*.py"))


def test_backend_does_not_import_concrete_insightface_model() -> None:
    backend_source = read_package("backend")

    assert "insightface_model" not in backend_source
    assert "InsightFaceModel" not in backend_source


def test_frontend_does_not_import_internal_src_packages() -> None:
    imports: list[str] = []
    for path in (ROOT / "src" / "frontend").glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.append(node.module)

    assert all(not name.startswith("src.") for name in imports)
