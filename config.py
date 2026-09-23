import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models" / "weights"
DATA_DIR = BASE_DIR / "data"
DATABASE_PATH = DATA_DIR / "signatures.db"
STATIC_DIR = BASE_DIR / "ui"
SAMPLES_DIR = BASE_DIR / "dataset" / "samples"

# Ensure directories exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

# Hugging Face Model Configuration
HF_REPO_ID = "Mels22/Signature-Detection-Verification"
DETECTOR_MODEL_FILE = "detector_yolo_1cls.pt"
VERIFIER_MODEL_FILE = "verifier_siamese.pt"

DETECTOR_WEIGHTS_PATH = MODELS_DIR / DETECTOR_MODEL_FILE
VERIFIER_WEIGHTS_PATH = MODELS_DIR / VERIFIER_MODEL_FILE

# Model Inference Configuration (Optimized for Pure CPU)
DEVICE = "cpu"
DETECTOR_IMG_SIZE = 768
DETECTOR_CONF_THRESHOLD = 0.35
DETECTOR_IOU_THRESHOLD = 0.45

VERIFIER_INPUT_SIZE = (105, 105)
VERIFIER_EMBEDDING_DIM = 256

# Verification Decision Thresholds (Calibrated for High Accuracy & Intra-Signer Pen Invariance)
# Distance is L2 Euclidean distance between L2-normalized 256-D vectors (range [0, 2])
# Cosine similarity = 1 - (L2^2) / 2
THRESHOLD_HIGH_MATCH = 0.82       # Euclidean distance <= 0.82 (Similarity >= 66%) -> VERIFIED_MATCH
THRESHOLD_PROBABLE_MATCH = 1.05   # Euclidean distance <= 1.05 (Similarity >= 45%) -> PROBABLE_MATCH (pen/surface variation)
THRESHOLD_INCONCLUSIVE = 1.25     # Euclidean distance <= 1.25 -> INCONCLUSIVE (Needs Manual Check)
# Euclidean distance > 1.25 -> REJECTED_FORGERY

# Server Configuration
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 8000
MCP_PORT = 8001
