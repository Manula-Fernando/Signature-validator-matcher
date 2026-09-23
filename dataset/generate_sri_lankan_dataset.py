import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import json
import logging

from config import SAMPLES_DIR, BASE_DIR
from core.database import SpecimenDatabase
from core.verifier import SignatureVerifier
from core.preprocessor import encode_image_base64

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("SriLankanDatasetGenerator")

def generate_sinhala_style_signature(seed: int = 42, pen_thickness: int = 2, slant: float = 0.0, pen_type: str = "ballpoint", is_forgery: bool = False) -> np.ndarray:
    """
    Synthesizes authentic Sri Lankan style signatures:
    - Genuine: Initial Sinhala loop ('Ma'/'Ka' script flourish) + cursive fluid surname + dynamic ribbon underline.
    - Forgery: Different stroke frequency, inverted ascender/descender kinematics, irregular loops (impostor attempt).
    """
    np.random.seed(seed)
    canvas = np.ones((180, 480, 3), dtype=np.uint8) * 255
    
    if is_forgery:
        # Forgery: Completely different handwriting kinematics and loop structures (e.g., Impostor signing differently)
        loop_pts = []
        for theta in np.linspace(0, 3.2 * np.pi, 140):
            r = 18 + 14 * np.sin(theta * 2.2)
            x = int(50 + r * 1.3 * np.cos(theta) + slant * 20)
            y = int(95 + r * np.sin(theta))
            loop_pts.append((x, y))

        t = np.linspace(0, 6 * np.pi, 280)
        sig_pts = []
        for i, val in enumerate(t):
            x = int(85 + i * 1.15 + 12 * np.cos(val * 2.5) + slant * 30)
            y = int(82 + 25 * np.cos(val * 1.2) * np.sin(val * 0.8) + 14 * np.sin(val * 4.2))
            sig_pts.append((x, y))

        und_pts = []
        for u in np.linspace(0, 1, 100):
            ux = int(60 + u * 350)
            uy = int(145 + 12 * np.sin(u * np.pi * 3.5))
            und_pts.append((ux, uy))
    else:
        # Genuine Signer: Characteristic smooth Sinhala initial loop + fluid ligature
        loop_pts = []
        for theta in np.linspace(0, 2.8 * np.pi, 120):
            r = 25 + 10 * np.cos(theta * 1.5)
            x = int(60 + r * np.cos(theta) + slant * 15)
            y = int(85 + r * 1.2 * np.sin(theta))
            loop_pts.append((x, y))

        t = np.linspace(0, 5 * np.pi, 280)
        sig_pts = []
        for i, val in enumerate(t):
            x = int(75 + i * 1.2 + 8 * np.sin(val * 1.8) + slant * 25)
            y = int(88 + 32 * np.sin(val) * np.cos(val * 0.45) + 8 * np.sin(val * 3.5))
            sig_pts.append((x, y))

        und_pts = []
        for u in np.linspace(0, 1, 100):
            ux = int(45 + u * 380)
            uy = int(140 + 8 * np.sin(u * np.pi * 2) - u * 15)
            und_pts.append((ux, uy))

    # Set ink color based on pen type
    if pen_type == "gel":
        ink_color = (25, 20, 18)      # Deep jet black
    elif pen_type == "fountain":
        ink_color = (130, 45, 20)     # Classic royal blue
    elif pen_type == "marker":
        ink_color = (30, 25, 25)      # Dense black marker
    else:  # ballpoint
        ink_color = (120, 50, 30)     # Ballpoint navy blue

    all_segments = [loop_pts, sig_pts, und_pts]
    for seg in all_segments:
        for j in range(len(seg) - 1):
            cv2.line(canvas, seg[j], seg[j+1], ink_color, pen_thickness, lineType=cv2.LINE_AA)

    # Double flourish line & terminal dots
    cv2.line(canvas, (330, 128), (430, 112), ink_color, pen_thickness, lineType=cv2.LINE_AA)
    cv2.circle(canvas, (438, 112), pen_thickness + 1, ink_color, -1)
    cv2.circle(canvas, (452, 110), pen_thickness + 1, ink_color, -1)

    return canvas

def overlay_signature(bg: np.ndarray, sig_img: np.ndarray, x: int, y: int, max_w: int, max_h: int) -> list:
    """Overlays signature ink naturally over background document pattern."""
    h, w = sig_img.shape[:2]
    scale = min(max_w / w, max_h / h)
    nw, nh = int(w * scale), int(h * scale)
    resized_sig = cv2.resize(sig_img, (nw, nh), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(resized_sig, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY_INV)

    roi = bg[y:y+nh, x:x+nw]
    for c in range(3):
        roi[:, :, c] = np.where(mask > 0, resized_sig[:, :, c], roi[:, :, c])
    bg[y:y+nh, x:x+nw] = roi
    return [x, y, x + nw, y + nh]

def build_sri_lankan_documents():
    """Builds authentic Sri Lankan banking and identification documents."""
    logger.info("Generating authentic Sri Lankan document templates...")

    # Generate Sinhala hybrid signatures for local customer
    sig_manula_ballpoint = generate_sinhala_style_signature(seed=555, pen_thickness=2, slant=0.0, pen_type="ballpoint")
    sig_manula_gel = generate_sinhala_style_signature(seed=555, pen_thickness=4, slant=0.03, pen_type="gel")
    sig_manula_fountain = generate_sinhala_style_signature(seed=555, pen_thickness=3, slant=-0.02, pen_type="fountain")
    sig_forged_manula = generate_sinhala_style_signature(seed=999, pen_thickness=2, slant=0.25, pen_type="ballpoint", is_forgery=True)

    # Save extra specimen files
    cv2.imwrite(str(SAMPLES_DIR / "manula_sig_ballpoint.png"), sig_manula_ballpoint)
    cv2.imwrite(str(SAMPLES_DIR / "manula_sig_gel.png"), sig_manula_gel)
    cv2.imwrite(str(SAMPLES_DIR / "manula_sig_fountain.png"), sig_manula_fountain)
    cv2.imwrite(str(SAMPLES_DIR / "manula_sig_forged.png"), sig_forged_manula)

    # -------------------------------------------------------------
    # 1. Sri Lankan Cheque: LOLC Finance PLC (Genuine & Forged)
    # -------------------------------------------------------------
    lolc_cheque = np.ones((480, 1120, 3), dtype=np.uint8) * 246
    # Light gold/amber security watermark tint (LOLC Corporate branding)
    lolc_cheque[:, :, 0] = 230  # B
    lolc_cheque[:, :, 1] = 242  # G
    lolc_cheque[:, :, 2] = 248  # R (warm cream/gold tint)

    # Guilloche border
    cv2.rectangle(lolc_cheque, (15, 15), (1105, 465), (160, 140, 90), 2)
    cv2.rectangle(lolc_cheque, (20, 20), (1100, 460), (200, 190, 150), 1)

    # Header - LOLC FINANCE PLC
    cv2.rectangle(lolc_cheque, (45, 35), (320, 85), (20, 50, 120), -1)  # LOLC Blue crest banner
    cv2.putText(lolc_cheque, "LOLC FINANCE PLC", (55, 70), cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 255, 255), 2)
    cv2.putText(lolc_cheque, "HEAD OFFICE / CORPORATE BRANCH - COLOMBO", (45, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (80, 70, 50), 1)
    cv2.putText(lolc_cheque, "No. 100/1, Sri Jayawardenepura Mawatha, Rajagiriya", (45, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (100, 90, 70), 1)

    # A/C Payee Only stamp (top left cross)
    cv2.line(lolc_cheque, (35, 145), (155, 35), (60, 60, 60), 2)
    cv2.line(lolc_cheque, (55, 155), (175, 45), (60, 60, 60), 2)
    cv2.putText(lolc_cheque, "A/C PAYEE ONLY", (55, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (50, 50, 50), 1)

    # Date Box (DD-MM-YYYY format typical of Sri Lankan banks)
    cv2.rectangle(lolc_cheque, (830, 40), (1070, 85), (140, 130, 90), 1)
    cv2.putText(lolc_cheque, "DATE:", (840, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 80), 1)
    cv2.putText(lolc_cheque, "2 2 - 0 9 - 2 0 2 6", (840, 78), cv2.FONT_HERSHEY_DUPLEX, 0.55, (30, 30, 30), 1)

    # Pay Line: Sinhala + English (Pay / හෝ දරන්නාට)
    cv2.putText(lolc_cheque, "PAY / හෝ දරන්නාට :", (45, 175), cv2.FONT_HERSHEY_DUPLEX, 0.58, (50, 40, 30), 1)
    cv2.putText(lolc_cheque, "FERNANDO LOGISTICS (PVT) LTD", (260, 175), cv2.FONT_HERSHEY_DUPLEX, 0.68, (20, 30, 80), 2)
    cv2.line(lolc_cheque, (250, 185), (820, 185), (120, 110, 90), 1)

    # Or Bearer
    cv2.putText(lolc_cheque, "OR BEARER", (840, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (90, 80, 60), 1)

    # Currency & Amount: "රුපියල් / RUPEES"
    cv2.putText(lolc_cheque, "RUPEES / රුපියල් :", (45, 235), cv2.FONT_HERSHEY_DUPLEX, 0.55, (50, 40, 30), 1)
    cv2.putText(lolc_cheque, "Four Hundred Eighty Five Thousand Rupees Only", (230, 235), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (20, 30, 80), 1)
    cv2.line(lolc_cheque, (220, 245), (1070, 245), (120, 110, 90), 1)

    # Amount box in LKR: "Rs. 485,000/="
    cv2.rectangle(lolc_cheque, (830, 145), (1070, 195), (140, 110, 40), 2)
    cv2.putText(lolc_cheque, "LKR  485,000/=", (845, 180), cv2.FONT_HERSHEY_DUPLEX, 0.72, (20, 30, 90), 2)

    # Account Number
    cv2.putText(lolc_cheque, "A/C NO: 0014-9932-1102", (45, 315), cv2.FONT_HERSHEY_DUPLEX, 0.65, (40, 40, 40), 1)
    cv2.putText(lolc_cheque, "ACCOUNT TITLE: K. A. MANULA FERNANDO", (45, 340), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 80), 1)

    # Signature box with bilingual title
    cv2.line(lolc_cheque, (720, 360), (1070, 360), (90, 90, 90), 2)
    cv2.putText(lolc_cheque, "AUTHORIZED SIGNATORY / බලයලත් අත්සන", (730, 385), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (90, 80, 70), 1)

    # Sri Lankan MICR Cheque band at bottom
    cv2.putText(lolc_cheque, "c 048192 c  7315 - 001 a  001499321102 c  25", (240, 442), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (30, 30, 30), 2)

    cheque_lolc_gen = lolc_cheque.copy()
    cheque_lolc_forg = lolc_cheque.copy()

    overlay_signature(cheque_lolc_gen, sig_manula_ballpoint, 750, 255, 300, 95)
    overlay_signature(cheque_lolc_forg, sig_forged_manula, 750, 255, 300, 95)

    # -------------------------------------------------------------
    # 2. Sri Lankan Smart National Identity Card (Smart NIC)
    # -------------------------------------------------------------
    sri_nic = np.ones((520, 820, 3), dtype=np.uint8) * 242
    sri_nic[:, :, 0] = 245
    sri_nic[:, :, 1] = 238
    sri_nic[:, :, 2] = 232

    cv2.rectangle(sri_nic, (20, 20), (800, 500), (150, 160, 180), 2)
    cv2.rectangle(sri_nic, (20, 20), (800, 90), (35, 65, 130), -1)
    cv2.putText(sri_nic, "DEMOCRATIC SOCIALIST REPUBLIC OF SRI LANKA", (150, 45), cv2.FONT_HERSHEY_DUPLEX, 0.58, (255, 255, 255), 1)
    cv2.putText(sri_nic, "NATIONAL IDENTITY CARD / ජාතික හැඳුනුම්පත", (170, 75), cv2.FONT_HERSHEY_DUPLEX, 0.54, (200, 225, 255), 1)

    cv2.rectangle(sri_nic, (50, 120), (220, 340), (180, 185, 195), -1)
    cv2.rectangle(sri_nic, (50, 120), (220, 340), (100, 110, 130), 2)
    cv2.putText(sri_nic, "HOLDER PHOTO", (65, 235), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 90, 110), 1)

    cv2.circle(sri_nic, (530, 220), 75, (220, 215, 200), 2)
    cv2.putText(sri_nic, "DEPARTMENT OF REGISTRATION OF PERSONS", (260, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (100, 100, 110), 1)

    cv2.putText(sri_nic, "NIC NO: 199815403219", (260, 160), cv2.FONT_HERSHEY_DUPLEX, 0.72, (20, 20, 30), 2)
    cv2.putText(sri_nic, "NAME: FERNANDO, K. A. MANULA", (260, 195), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 30, 40), 1)
    cv2.putText(sri_nic, "DOB: 1998-06-02    SEX: MALE", (260, 225), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (50, 50, 60), 1)
    cv2.putText(sri_nic, "PLACE OF BIRTH: COLOMBO", (260, 255), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (50, 50, 60), 1)
    cv2.putText(sri_nic, "ADDRESS: 45/2, TEMPLE ROAD, NAWALA", (260, 285), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (50, 50, 60), 1)

    cv2.rectangle(sri_nic, (260, 330), (590, 440), (170, 180, 200), 1)
    cv2.putText(sri_nic, "HOLDER'S SIGNATURE / හිමිකරුගේ අත්සන", (270, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (90, 100, 120), 1)
    overlay_signature(sri_nic, sig_manula_gel, 275, 345, 300, 85)

    cv2.rectangle(sri_nic, (640, 320), (760, 440), (60, 60, 60), 2)
    cv2.putText(sri_nic, "SECURE", (670, 370), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (60, 60, 60), 1)
    cv2.putText(sri_nic, "QR CODE", (665, 395), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (60, 60, 60), 1)

    # -------------------------------------------------------------
    # 3. Sri Lankan Driving License (DMT)
    # -------------------------------------------------------------
    dmt_card = np.ones((500, 800, 3), dtype=np.uint8) * 238
    dmt_card[:, :, 1] = 244
    cv2.rectangle(dmt_card, (20, 20), (780, 480), (120, 160, 140), 2)
    cv2.rectangle(dmt_card, (20, 20), (780, 85), (20, 90, 70), -1)
    cv2.putText(dmt_card, "DRIVING LICENCE - SRI LANKA", (220, 50), cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 255), 1)
    cv2.putText(dmt_card, "DEPARTMENT OF MOTOR TRAFFIC / මෝටර් රථ ප්‍රවාහන", (200, 73), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 240, 220), 1)

    cv2.rectangle(dmt_card, (45, 110), (200, 310), (180, 190, 185), -1)
    cv2.putText(dmt_card, "PHOTO", (90, 215), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (100, 110, 105), 1)

    sig_bandara_ballpoint = generate_sinhala_style_signature(seed=777, pen_thickness=2, slant=-0.05, pen_type="ballpoint")
    sig_bandara_gel = generate_sinhala_style_signature(seed=777, pen_thickness=4, slant=-0.04, pen_type="gel")
    cv2.imwrite(str(SAMPLES_DIR / "bandara_sig_ballpoint.png"), sig_bandara_ballpoint)
    cv2.imwrite(str(SAMPLES_DIR / "bandara_sig_gel.png"), sig_bandara_gel)

    cv2.putText(dmt_card, "LICENCE NO: B3982145", (240, 130), cv2.FONT_HERSHEY_DUPLEX, 0.7, (20, 40, 30), 2)
    cv2.putText(dmt_card, "NAME: BANDARA, CHAMINDA", (240, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (30, 40, 30), 1)
    cv2.putText(dmt_card, "NIC NO: 198514902148", (240, 195), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 50, 40), 1)
    cv2.putText(dmt_card, "VEHICLE CLASSES: A, B1, B (AUTO/MANUAL)", (240, 225), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (40, 50, 40), 1)
    cv2.putText(dmt_card, "DATE OF EXPIRY: 2032-11-15", (240, 255), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (40, 50, 40), 1)

    cv2.rectangle(dmt_card, (240, 300), (530, 400), (160, 190, 175), 1)
    overlay_signature(dmt_card, sig_bandara_gel, 250, 315, 275, 80)

    # -------------------------------------------------------------
    # 4. Commercial Lease / Facility Agreement (LOLC Finance PLC)
    # -------------------------------------------------------------
    lease = np.ones((840, 640, 3), dtype=np.uint8) * 252
    cv2.rectangle(lease, (20, 20), (620, 820), (100, 100, 100), 1)
    cv2.putText(lease, "LOLC FINANCE PLC", (180, 55), cv2.FONT_HERSHEY_DUPLEX, 0.75, (20, 50, 130), 2)
    cv2.putText(lease, "COMMERCIAL VEHICLE LEASE / FACILITY AGREEMENT", (90, 85), cv2.FONT_HERSHEY_DUPLEX, 0.52, (40, 40, 40), 1)
    cv2.line(lease, (40, 98), (600, 98), (120, 120, 120), 1)

    cv2.rectangle(lease, (480, 115), (580, 215), (150, 60, 40), 2)
    cv2.putText(lease, "STAMP DUTY", (490, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 60, 40), 1)
    cv2.putText(lease, "RS. 250/=", (500, 185), cv2.FONT_HERSHEY_DUPLEX, 0.52, (150, 60, 40), 1)

    lease_clauses = [
        "Agreement Reference: LOLC-FL-2026-88912",
        "Date of Execution: 22nd September 2026",
        "Lender: LOLC Finance PLC, Rajagiriya, Sri Lanka",
        "Customer: K. A. Manula Fernando (NIC: 199815403219)",
        "Facility Amount: LKR 3,500,000.00 (Sri Lankan Rupees)",
        "Asset: Toyota Hilux Double Cab (WP - CAA 9421)",
        "Tenor: 48 Monthly Installments @ CBSL Benchmark Rate",
        "",
        "The Lessee hereby accepts the delivery of the leased vehicle",
        "and agrees to pay all installments on or before the due date.",
        "Default in two consecutive payments gives the Lessor immediate",
        "right of repossession under the Finance Leasing Act No. 56 of 2000.",
        "",
        "IN WITNESS WHEREOF the parties have set their hands hereunto:"
    ]

    for idx, cl in enumerate(lease_clauses):
        cv2.putText(lease, cl, (45, 130 + idx * 26), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (40, 40, 40), 1)

    cv2.line(lease, (50, 710), (270, 710), (90, 90, 90), 1)
    cv2.putText(lease, "AUTHORISED SIGNATORY (LOLC)", (60, 735), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (90, 90, 90), 1)

    cv2.line(lease, (340, 710), (580, 710), (90, 90, 90), 1)
    cv2.putText(lease, "LESSEE / CUSTOMER SIGNATURE", (355, 735), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (90, 90, 90), 1)

    overlay_signature(lease, sig_manula_fountain, 350, 630, 220, 75)

    new_docs = {
        "cheque_lolc_finance_genuine.jpg": cheque_lolc_gen,
        "cheque_lolc_finance_forged.jpg": cheque_lolc_forg,
        "sri_lanka_smart_nic.jpg": sri_nic,
        "sri_lanka_driving_license.jpg": dmt_card,
        "lolc_lease_agreement.jpg": lease
    }

    for fname, img in new_docs.items():
        p = SAMPLES_DIR / fname
        cv2.imwrite(str(p), img)
        logger.info(f"Saved Sri Lankan document: {fname}")

def enroll_expanded_customers():
    """Enrolls Sri Lankan and international customers into SQLite specimen store."""
    logger.info("Enrolling expanded customer base into SQLite database...")
    db = SpecimenDatabase()
    verifier = SignatureVerifier()

    # Clear outdated vector specimens
    with db._get_connection() as conn:
        conn.execute("DELETE FROM specimens")
        conn.execute("DELETE FROM customers")
        conn.execute("DELETE FROM audit_logs")
        conn.commit()

    expanded_customers = [
        {
            "id": "CUST_001",
            "name": "Johnathan Carter",
            "account": "0092-4821-9941",
            "doc_type": "Passport",
            "specimen_files": ["person_A_sig1.png", "person_A_sig2.png", "person_A_sig3.png", "person_A_sig4.png"]
        },
        {
            "id": "CUST_002",
            "name": "Manula Fernando",
            "account": "0014-9932-1102",
            "doc_type": "Sri Lankan Smart NIC (199815403219)",
            "specimen_files": ["manula_sig_ballpoint.png", "manula_sig_gel.png", "manula_sig_fountain.png"]
        },
        {
            "id": "CUST_003",
            "name": "Chaminda Bandara",
            "account": "0025-8814-3390",
            "doc_type": "Driving License (B3982145)",
            "specimen_files": ["bandara_sig_ballpoint.png", "bandara_sig_gel.png"]
        },
        {
            "id": "CUST_004",
            "name": "Dilani Perera",
            "account": "0071-6623-4411",
            "doc_type": "Sri Lankan NIC (198762104598)",
            "specimen_files": ["person_D_sig1.png", "person_D_sig2.png", "person_D_sig3.png"]
        },
        {
            "id": "CUST_005",
            "name": "Sarah Jenkins",
            "account": "0088-7741-3320",
            "doc_type": "International Passport",
            "specimen_files": ["person_E_sig1.png", "person_E_sig2.png"]
        }
    ]

    for c in expanded_customers:
        db.enroll_customer(c["id"], c["name"], c["account"], c["doc_type"])
        for s_file in c["specimen_files"]:
            s_path = SAMPLES_DIR / s_file
            if s_path.exists():
                img = cv2.imread(str(s_path))
                b64 = encode_image_base64(img, format_ext=".png")
                emb, _, _ = verifier.extract_embedding(img)
                db.add_specimen(c["id"], b64, emb)

    logger.info("Successfully enrolled 5 customer profiles with multiple pen specimens.")

def update_index_metadata():
    """Updates sample index.json for UI dropdown and testing suite."""
    sample_docs_meta = [
        {
            "id": "cheque_lolc_finance_genuine",
            "title": "Sri Lankan Cheque: LOLC Finance (Genuine)",
            "description": "Authentic LOLC Finance PLC cheque in LKR signed by Manula Fernando (CUST_002).",
            "filename": "cheque_lolc_finance_genuine.jpg",
            "expected_customer_id": "CUST_002",
            "expected_verdict": "VERIFIED_MATCH"
        },
        {
            "id": "cheque_lolc_finance_forged",
            "title": "Sri Lankan Cheque: LOLC Finance (Forged)",
            "description": "Fraudulent LOLC Finance cheque with attempted forgery drawn on Manula Fernando's account.",
            "filename": "cheque_lolc_finance_forged.jpg",
            "expected_customer_id": "CUST_002",
            "expected_verdict": "REJECTED_FORGERY"
        },
        {
            "id": "sri_lanka_smart_nic",
            "title": "Sri Lankan Smart NIC (Polycarbonate)",
            "description": "Department of Registration of Persons Smart NIC (12-digit: 199815403219) with holder signature.",
            "filename": "sri_lanka_smart_nic.jpg",
            "expected_customer_id": "CUST_002",
            "expected_verdict": "VERIFIED_MATCH"
        },
        {
            "id": "sri_lanka_driving_license",
            "title": "Sri Lankan Driving License (DMT)",
            "description": "Department of Motor Traffic official driving card belonging to Chaminda Bandara (CUST_003).",
            "filename": "sri_lanka_driving_license.jpg",
            "expected_customer_id": "CUST_003",
            "expected_verdict": "VERIFIED_MATCH"
        },
        {
            "id": "lolc_lease_agreement",
            "title": "LOLC Finance Lease / Facility Agreement",
            "description": "Official commercial vehicle lease agreement executed by Manula Fernando (CUST_002).",
            "filename": "lolc_lease_agreement.jpg",
            "expected_customer_id": "CUST_002",
            "expected_verdict": "VERIFIED_MATCH"
        },
        {
            "id": "cheque_genuine",
            "title": "Global Bank Cheque (Genuine)",
            "description": "Standard Global Trust Bank cheque signed by customer Johnathan Carter (CUST_001).",
            "filename": "cheque_genuine.jpg",
            "expected_customer_id": "CUST_001",
            "expected_verdict": "VERIFIED_MATCH"
        },
        {
            "id": "cheque_forged",
            "title": "Global Bank Cheque (Forged)",
            "description": "Fraudulent cheque drawn on Johnathan Carter's account with a forged signature.",
            "filename": "cheque_forged.jpg",
            "expected_customer_id": "CUST_001",
            "expected_verdict": "REJECTED_FORGERY"
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
            "id": "national_id",
            "title": "National ID Card (General)",
            "description": "National Identity Card specimen with Johnathan Carter signature.",
            "filename": "national_id_card.jpg",
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
            {"filename": "manula_sig_ballpoint.png", "signer": "Manula Fernando (CUST_002 - Ballpoint Pen)"},
            {"filename": "manula_sig_gel.png", "signer": "Manula Fernando (CUST_002 - Thick Gel Pen)"},
            {"filename": "manula_sig_fountain.png", "signer": "Manula Fernando (CUST_002 - Fountain Pen)"},
            {"filename": "manula_sig_forged.png", "signer": "Attempted Forgery (Impostor on CUST_002)"},
            {"filename": "person_A_sig1.png", "signer": "Johnathan Carter (CUST_001 - Specimen 1)"},
            {"filename": "person_A_sig2.png", "signer": "Johnathan Carter (CUST_001 - Specimen 2)"},
            {"filename": "person_C_sig1.png", "signer": "Chaminda Bandara (CUST_003 - Specimen 1)"},
            {"filename": "person_D_sig1.png", "signer": "Dilani Perera (CUST_004 - Specimen 1)"},
            {"filename": "person_E_sig1.png", "signer": "Sarah Jenkins (CUST_005 - Specimen 1)"}
        ]
    }

    index_file = SAMPLES_DIR / "index.json"
    with open(index_file, "w", encoding="utf-8") as f:
        json.dump(index_data, f, indent=2)

    logger.info(f"Updated index.json successfully at {index_file}")

if __name__ == "__main__":
    build_sri_lankan_documents()
    enroll_expanded_customers()
    update_index_metadata()
    print("Sri Lankan Document and Specimen Generation Completed!")
