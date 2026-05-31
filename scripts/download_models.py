from pathlib import Path

from huggingface_hub import snapshot_download

from src.common.logging import get_logger
from src.common.retry import with_retry


logger = get_logger(__name__)
HF_REPO_ID = "TODO_org/face-recognition-assignment-models"
LOCAL_DIR = Path("models")


@with_retry(max_attempts=3, base_delay=1.0, exceptions=(OSError, TimeoutError, ConnectionError))
def download_snapshot() -> None:
    # TODO: Create the HF repository and upload any project-owned files larger
    # than 50MB there. Runtime downloads should place them under models/.
    if HF_REPO_ID.startswith("TODO_"):
        logger.warning("HF_REPO_ID 仍是 TODO，占位脚本不会下载文件")
        LOCAL_DIR.mkdir(exist_ok=True)
        return
    snapshot_download(repo_id=HF_REPO_ID, local_dir=LOCAL_DIR)


if __name__ == "__main__":
    download_snapshot()
