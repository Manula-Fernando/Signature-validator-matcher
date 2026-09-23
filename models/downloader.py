import os
import sys
import shutil
import logging
from pathlib import Path
from huggingface_hub import hf_hub_download

# Support running directly or as module
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    HF_REPO_ID,
    DETECTOR_MODEL_FILE,
    VERIFIER_MODEL_FILE,
    DETECTOR_WEIGHTS_PATH,
    VERIFIER_WEIGHTS_PATH,
    MODELS_DIR
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ModelDownloader")

def download_models(force: bool = False):
    """
    Downloads pretrained signature detector (YOLO11s) and verifier (Siamese CNN)
    from Hugging Face repository locally to models/weights/ directory.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Download Detector Model
    if not DETECTOR_WEIGHTS_PATH.exists() or force:
        logger.info(f"Downloading detector model ({DETECTOR_MODEL_FILE}) from {HF_REPO_ID}...")
        downloaded_detector = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=DETECTOR_MODEL_FILE,
            local_dir=str(MODELS_DIR),
            local_dir_use_symlinks=False
        )
        logger.info(f"Detector model downloaded to {downloaded_detector}")
    else:
        logger.info(f"Detector model already exists at {DETECTOR_WEIGHTS_PATH}")
        
    # 2. Download Verifier Model
    if not VERIFIER_WEIGHTS_PATH.exists() or force:
        logger.info(f"Downloading verifier model ({VERIFIER_MODEL_FILE}) from {HF_REPO_ID}...")
        downloaded_verifier = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=VERIFIER_MODEL_FILE,
            local_dir=str(MODELS_DIR),
            local_dir_use_symlinks=False
        )
        logger.info(f"Verifier model downloaded to {downloaded_verifier}")
    else:
        logger.info(f"Verifier model already exists at {VERIFIER_WEIGHTS_PATH}")

    det_size_mb = DETECTOR_WEIGHTS_PATH.stat().st_size / (1024 * 1024)
    ver_size_mb = VERIFIER_WEIGHTS_PATH.stat().st_size / (1024 * 1024)
    logger.info(f"Models verification complete: Detector ({det_size_mb:.2f} MB), Verifier ({ver_size_mb:.2f} MB)")
    return {
        "detector_path": str(DETECTOR_WEIGHTS_PATH),
        "verifier_path": str(VERIFIER_WEIGHTS_PATH),
        "detector_size_mb": det_size_mb,
        "verifier_size_mb": ver_size_mb
    }

if __name__ == "__main__":
    download_models()
