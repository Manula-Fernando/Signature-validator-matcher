import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import json
import logging
from huggingface_hub import hf_hub_download

from config import SAMPLES_DIR, BASE_DIR
from core.database import SpecimenDatabase
from core.verifier import SignatureVerifier
from core.preprocessor import encode_image_base64

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("SampleGenerator")

HF_DATASET_REPO = "rakshitdabral/Signature-Verification-Dataset"

def download_raw_signatures():
    """
    Downloads authentic test signatures from Hugging Face dataset.
    """
    logger.info(f"Downloading authentic signature samples from {HF_DATASET_REPO}...")
    sample_files = [
        # Person 283 (Signer A)
        ("dataset/test/person_0283/sig16.png", "person_A_sig1.png"),
        ("dataset/test/person_0283/sig17.png", "person_A_sig2.png"),
        ("dataset/test/person_0283/sig18.png", "person_A_sig3.png"),
        # Person 284 (Signer B)
        ("dataset/test/person_0284/sig16.png", "person_B_sig1.png"),
        ("dataset/test/person_0284/sig17.png", "person_B_sig2.png"),
        # Person 285 (Signer C)
        ("dataset/test/person_0285/sig16.png", "person_C_sig1.png"),
    ]
    
    downloaded = {}
    for remote_path, local_name in sample_files:
        local_path = SAMPLES_DIR / local_name
        if not local_path.exists():
            try:
                p = hf_hub_download(
                    repo_id=HF_DATASET_REPO,
                    filename=remote_path,
                    repo_type="dataset",
                    local_dir=str(SAMPLES_DIR),
                    local_dir_use_symlinks=False
                )
                # Move to standard name
                p_path = Path(p)
                if p_path.exists() and p_path != local_path:
                    local_path.write_bytes(p_path.read_bytes())
                downloaded[local_name] = local_path
            except Exception as e:
                logger.warning(f"Failed to download {remote_path}: {e}")
        else:
            downloaded[local_name] = local_path
            
    return downloaded

def create_synthetic_signature(name: str, pen_thickness: int = 2, slant: float = 0.0) -> np.ndarray:
    """
    Creates realistic signature stroke geometry for testing pen variation if offline.
    """
    canvas = np.ones((160, 420, 3), dtype=np.uint8) * 255
    # Generate signature loops and strokes
    pts = []
    t = np.linspace(0, 4 * np.pi, 250)
    for i, val in enumerate(t):
        x = int(30 + i * 1.4 + 10 * np.sin(val * 2) + slant * 20)
        y = int(80 + 35 * np.sin(val) * np.cos(val * 0.5) + 10 * np.sin(val * 4))
        pts.append((x, y))

    for j in range(len(pts) - 1):
        color = (35, 30, 25)  # dark ink
        cv2.line(canvas, pts[j], pts[j+1], color, pen_thickness, lineType=cv2.LINE_AA)
        
    # Add stylish underline flourish
    cv2.line(canvas, (25, 125), (380, 115), (35, 30, 25), pen_thickness, lineType=cv2.LINE_AA)
    cv2.line(canvas, (320, 115), (395, 95), (35, 30, 25), pen_thickness, lineType=cv2.LINE_AA)
    
    return canvas

def generate_document_templates():
    """
    Generates high-resolution, realistic document templates:
    1. Bank Cheque
    2. National Identity Card (NIC)
    3. Official Passport Page
    4. Commercial Loan Contract
    """
    logger.info("Generating realistic document templates (Cheque, NIC, Passport, Contract)...")

    # Load authentic signatures
    sig_a1_path = SAMPLES_DIR / "person_A_sig1.png"
    sig_a2_path = SAMPLES_DIR / "person_A_sig2.png"
    sig_b1_path = SAMPLES_DIR / "person_B_sig1.png"

    if sig_a1_path.exists():
        sig_a1 = cv2.imread(str(sig_a1_path))
    else:
        sig_a1 = create_synthetic_signature("J. Carter", pen_thickness=2)

    if sig_a2_path.exists():
        sig_a2 = cv2.imread(str(sig_a2_path))
    else:
        sig_a2 = create_synthetic_signature("J. Carter (Thick)", pen_thickness=4)

    if sig_b1_path.exists():
        sig_forged = cv2.imread(str(sig_b1_path))
    else:
        sig_forged = create_synthetic_signature("Forged J. Carter", pen_thickness=2, slant=0.5)

    def overlay_signature(bg, sig_img, x, y, max_w, max_h):
        # Resize signature to fit region
        h, w = sig_img.shape[:2]
        scale = min(max_w / w, max_h / h)
        nw, nh = int(w * scale), int(h * scale)
        resized_sig = cv2.resize(sig_img, (nw, nh), interpolation=cv2.INTER_AREA)

        # Convert to gray for mask
        gray = cv2.cvtColor(resized_sig, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY_INV)

        roi = bg[y:y+nh, x:x+nw]
        # Blend signature ink onto background
        for c in range(3):
            roi[:, :, c] = np.where(mask > 0, resized_sig[:, :, c], roi[:, :, c])
        bg[y:y+nh, x:x+nw] = roi
        return [x, y, x + nw, y + nh]

    # --- 1. Bank Cheque ---
    cheque = np.ones((480, 1100, 3), dtype=np.uint8) * 245
    # Light security watermark tint
    cheque[:, :, 0] = 238
    cheque[:, :, 1] = 245
    cheque[:, :, 2] = 240
    # Cheque border & header
    cv2.rectangle(cheque, (20, 20), (1080, 460), (160, 180, 160), 2)
    cv2.putText(cheque, "GLOBAL TRUST BANK PLC", (60, 65), cv2.FONT_HERSHEY_DUPLEX, 0.9, (40, 80, 40), 2)
    cv2.putText(cheque, "BRANCH: CORPORATE HEADQUARTERS", (60, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 120, 100), 1)
    
    # Date box
    cv2.rectangle(cheque, (820, 45), (1040, 90), (140, 160, 140), 1)
    cv2.putText(cheque, "DATE: 2026-09-22", (835, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 50), 1)

    # Pay line
    cv2.putText(cheque, "PAY TO THE ORDER OF:", (60, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (70, 70, 70), 1)
    cv2.putText(cheque, "ACME INDUSTRIAL LOGISTICS CORP.", (300, 160), cv2.FONT_HERSHEY_DUPLEX, 0.7, (20, 20, 80), 2)
    cv2.line(cheque, (280, 170), (780, 170), (120, 120, 120), 1)

    # Amount box
    cv2.rectangle(cheque, (820, 140), (1040, 185), (40, 80, 40), 2)
    cv2.putText(cheque, "$ 125,450.00", (840, 172), cv2.FONT_HERSHEY_DUPLEX, 0.75, (20, 20, 80), 2)

    # Amount in words
    cv2.putText(cheque, "THE SUM OF:", (60, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (70, 70, 70), 1)
    cv2.putText(cheque, "One Hundred Twenty Five Thousand Four Hundred Fifty Dollars Only", (200, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 80), 1)
    cv2.line(cheque, (180, 240), (1040, 240), (120, 120, 120), 1)

    # Account Number
    cv2.putText(cheque, "A/C NO: 0092-4821-9941", (60, 310), cv2.FONT_HERSHEY_DUPLEX, 0.65, (50, 50, 50), 1)

    # Signature box & line
    cv2.line(cheque, (720, 360), (1040, 360), (80, 80, 80), 2)
    cv2.putText(cheque, "AUTHORIZED SIGNATURE", (770, 385), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 100, 100), 1)

    # MICR line at bottom
    cv2.putText(cheque, "C104928C  019283471A  009248219941C  01", (280, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (30, 30, 30), 2)

    # Clone for genuine vs forged cheque
    cheque_genuine = cheque.copy()
    cheque_forged = cheque.copy()

    # Overlay signatures in signature zone (730, 260)
    overlay_signature(cheque_genuine, sig_a1, 740, 250, 280, 100)
    overlay_signature(cheque_forged, sig_forged, 740, 250, 280, 100)

    # --- 2. National Identity Card (NIC) ---
    nic = np.ones((500, 800, 3), dtype=np.uint8) * 235
    # Card border
    cv2.rectangle(nic, (20, 20), (780, 480), (180, 180, 200), 2)
    # Header
    cv2.rectangle(nic, (20, 20), (780, 90), (30, 60, 120), -1)
    cv2.putText(nic, "DEMOCRATIC SOCIALIST REPUBLIC", (180, 50), cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 255), 1)
    cv2.putText(nic, "NATIONAL IDENTITY CARD", (250, 75), cv2.FONT_HERSHEY_DUPLEX, 0.6, (200, 220, 255), 1)

    # Photo placeholder
    cv2.rectangle(nic, (50, 120), (220, 330), (160, 160, 170), -1)
    cv2.putText(nic, "PHOTO", (100, 230), cv2.FONT_HERSHEY_DUPLEX, 0.7, (230, 230, 230), 1)

    # Details
    cv2.putText(nic, "ID NO: 198829401928", (260, 140), cv2.FONT_HERSHEY_DUPLEX, 0.7, (20, 20, 20), 2)
    cv2.putText(nic, "NAME: CARTER, JOHNATHAN", (260, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 40, 40), 1)
    cv2.putText(nic, "DOB: 1988-04-12", (260, 215), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60, 60, 60), 1)
    cv2.putText(nic, "SEX: MALE", (260, 245), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60, 60, 60), 1)
    cv2.putText(nic, "ADDRESS: 42 PARK AVENUE, NY", (260, 275), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60, 60, 60), 1)

    # Signature box on NIC
    cv2.rectangle(nic, (260, 320), (560, 420), (200, 200, 210), 1)
    cv2.putText(nic, "HOLDER'S SIGNATURE", (270, 340), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1)
    overlay_signature(nic, sig_a2, 270, 330, 270, 80)

    # --- 3. Official Passport Page ---
    passport = np.ones((650, 950, 3), dtype=np.uint8) * 242
    cv2.rectangle(passport, (25, 25), (925, 625), (120, 80, 60), 2)
    cv2.putText(passport, "PASSPORT / PASSEPORT", (320, 65), cv2.FONT_HERSHEY_DUPLEX, 0.8, (60, 40, 30), 2)

    # Photo Box
    cv2.rectangle(passport, (60, 100), (280, 380), (180, 180, 190), -1)
    cv2.putText(passport, "OFFICIAL PHOTO", (90, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (230, 230, 230), 1)

    # Passport details
    cv2.putText(passport, "PASSPORT NO: N9823145", (320, 120), cv2.FONT_HERSHEY_DUPLEX, 0.7, (20, 20, 20), 2)
    cv2.putText(passport, "SURNAME: FERNANDO", (320, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 40, 40), 1)
    cv2.putText(passport, "GIVEN NAMES: MANULA", (320, 195), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 40, 40), 1)
    cv2.putText(passport, "NATIONALITY: SRI LANKAN", (320, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (50, 50, 50), 1)
    cv2.putText(passport, "DATE OF ISSUE: 15 JAN 2024", (320, 265), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (50, 50, 50), 1)

    # Passport Signature Zone
    cv2.line(passport, (320, 350), (680, 350), (140, 140, 140), 1)
    cv2.putText(passport, "SIGNATURE OF BEARER", (420, 370), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
    overlay_signature(passport, sig_a1, 340, 265, 300, 80)

    # MRZ lines
    cv2.putText(passport, "P<LKAFERNANDO<<MANULA<<<<<<<<<<<<<<<<<<<<<<<", (60, 550), cv2.FONT_HERSHEY_PLAIN, 1.3, (30, 30, 30), 2)
    cv2.putText(passport, "N9823145<4LKA9508124M3401158<<<<<<<<<<<<<<02", (60, 590), cv2.FONT_HERSHEY_PLAIN, 1.3, (30, 30, 30), 2)

    # --- 4. Legal Loan Agreement / Contract ---
    contract = np.ones((800, 620, 3), dtype=np.uint8) * 252
    cv2.putText(contract, "COMMERCIAL LOAN AGREEMENT", (90, 70), cv2.FONT_HERSHEY_DUPLEX, 0.75, (20, 20, 20), 2)
    cv2.line(contract, (50, 85), (570, 85), (100, 100, 100), 1)

    lines = [
        "This Commercial Loan Agreement is entered into on 2026-09-22",
        "between GLOBAL TRUST BANK PLC ('Lender') and BORROWER.",
        "The Borrower acknowledges receipt of financing under the terms",
        "and covenants described herein. Repayment shall occur in",
        "quarterly installments with interest rate benchmarked to SOFR.",
        "In witness whereof, the parties hereto have executed this",
        "Agreement as of the date first above written."
    ]
    for i, line in enumerate(lines):
        cv2.putText(contract, line, (50, 130 + i * 35), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (60, 60, 60), 1)

    # Signature blocks
    cv2.line(contract, (60, 680), (280, 680), (100, 100, 100), 1)
    cv2.putText(contract, "BANK REPRESENTATIVE", (80, 705), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)

    cv2.line(contract, (340, 680), (560, 680), (100, 100, 100), 1)
    cv2.putText(contract, "BORROWER SIGNATURE", (360, 705), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)

    overlay_signature(contract, sig_a1, 350, 600, 200, 75)

    # Save all documents
    docs = {
        "cheque_genuine.jpg": cheque_genuine,
        "cheque_forged.jpg": cheque_forged,
        "national_id_card.jpg": nic,
        "passport_page.jpg": passport,
        "loan_contract.jpg": contract
    }

    doc_paths = {}
    for filename, img in docs.items():
        p = SAMPLES_DIR / filename
        cv2.imwrite(str(p), img)
        doc_paths[filename] = p

    logger.info("Sample documents created successfully.")
    return doc_paths

def populate_database_and_index():
    """
    Populates sample customers in the database with reference specimens
    and exports samples/index.json for the API and Web UI.
    """
    logger.info("Populating customer specimens in SQLite database...")
    db = SpecimenDatabase()
    verifier = SignatureVerifier()

    # Pre-register 3 Enterprise Customers
    customers = [
        {
            "id": "CUST_001",
            "name": "Johnathan Carter",
            "account": "0092-4821-9941",
            "doc_type": "Passport",
            "specimen_files": ["person_A_sig1.png", "person_A_sig2.png", "person_A_sig3.png"]
        },
        {
            "id": "CUST_002",
            "name": "Manula Fernando",
            "account": "0014-9932-1102",
            "doc_type": "NIC",
            "specimen_files": ["person_B_sig1.png", "person_B_sig2.png"]
        },
        {
            "id": "CUST_003",
            "name": "Sarah Jenkins",
            "account": "0088-7741-3320",
            "doc_type": "Driving License",
            "specimen_files": ["person_C_sig1.png"]
        }
    ]

    for c in customers:
        db.enroll_customer(c["id"], c["name"], c["account"], c["doc_type"])
        for s_file in c["specimen_files"]:
            s_path = SAMPLES_DIR / s_file
            if s_path.exists():
                img = cv2.imread(str(s_path))
                b64 = encode_image_base64(img, format_ext=".png")
                emb, _, _ = verifier.extract_embedding(img)
                db.add_specimen(c["id"], b64, emb)

    # Build index.json metadata
    sample_docs_meta = [
        {
            "id": "cheque_genuine",
            "title": "Bank Cheque (Genuine Signer)",
            "description": "Standard Global Trust Bank cheque signed by customer Johnathan Carter (CUST_001).",
            "filename": "cheque_genuine.jpg",
            "expected_customer_id": "CUST_001",
            "expected_verdict": "VERIFIED_MATCH"
        },
        {
            "id": "cheque_forged",
            "title": "Bank Cheque (Forged Signer)",
            "description": "Fraudulent cheque drawn on Johnathan Carter's account with a forged signature.",
            "filename": "cheque_forged.jpg",
            "expected_customer_id": "CUST_001",
            "expected_verdict": "REJECTED_FORGERY"
        },
        {
            "id": "national_id",
            "title": "National Identity Card (NIC)",
            "description": "Official NIC card with dedicated signature specimen region.",
            "filename": "national_id_card.jpg",
            "expected_customer_id": "CUST_001",
            "expected_verdict": "VERIFIED_MATCH"
        },
        {
            "id": "passport_page",
            "title": "Official Passport Page",
            "description": "International passport data page with Bearer's signature.",
            "filename": "passport_page.jpg",
            "expected_customer_id": "CUST_001",
            "expected_verdict": "VERIFIED_MATCH"
        },
        {
            "id": "loan_contract",
            "title": "Commercial Loan Contract",
            "description": "Legal contract document with borrower signature line at bottom.",
            "filename": "loan_contract.jpg",
            "expected_customer_id": "CUST_001",
            "expected_verdict": "VERIFIED_MATCH"
        }
    ]

    index_data = {
        "documents": sample_docs_meta,
        "sample_signatures": [
            {"filename": "person_A_sig1.png", "signer": "Johnathan Carter (Specimen 1 - Ballpoint)"},
            {"filename": "person_A_sig2.png", "signer": "Johnathan Carter (Specimen 2 - Gel Pen)"},
            {"filename": "person_A_sig3.png", "signer": "Johnathan Carter (Specimen 3 - Felt Tip)"},
            {"filename": "person_B_sig1.png", "signer": "Manula Fernando (Specimen 1)"},
            {"filename": "person_B_sig2.png", "signer": "Manula Fernando (Specimen 2)"},
            {"filename": "person_C_sig1.png", "signer": "Sarah Jenkins (Specimen 1)"}
        ]
    }

    index_file = SAMPLES_DIR / "index.json"
    with open(index_file, "w", encoding="utf-8") as f:
        json.dump(index_data, f, indent=2)

    logger.info(f"Sample index saved to {index_file}")

if __name__ == "__main__":
    download_raw_signatures()
    generate_document_templates()
    populate_database_and_index()
    logger.info("Sample generation and database population completed successfully.")
