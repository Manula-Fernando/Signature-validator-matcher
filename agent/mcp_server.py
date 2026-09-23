import logging
import json
from pathlib import Path
from typing import List, Optional
from fastmcp import FastMCP

from config import MCP_PORT
from core.detector import SignatureDetector
from core.verifier import SignatureVerifier
from core.database import SpecimenDatabase

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("SignatureAgentMCP")

# Initialize FastMCP Server
mcp = FastMCP(
    "SignatureVerificationAgent",
    instructions="Enterprise AI Agent Tool for detecting handwritten signatures in documents (cheques, IDs, passports, contracts) and verifying them against reference customer specimen signatures."
)

# Global instances
detector = None
verifier = None
db = None

def get_engines():
    global detector, verifier, db
    if detector is None:
        detector = SignatureDetector()
    if verifier is None:
        verifier = SignatureVerifier()
    if db is None:
        db = SpecimenDatabase()
    return detector, verifier, db

@mcp.tool()
def detect_signatures(document_path_or_base64: str, confidence_threshold: float = 0.35) -> str:
    """
    Detects and localizes handwritten signature regions within a document image
    (such as a bank cheque, national ID card, passport, or legal contract).
    Returns bounding box coordinates [x1, y1, x2, y2], detection count, and confidence scores.
    """
    det, _, _ = get_engines()
    res = det.detect(document_path_or_base64, conf_threshold=confidence_threshold)
    output = {
        "detected": res["detected"],
        "signature_count": res["count"],
        "detections": [
            {
                "id": s["id"],
                "bbox": s["bbox"],
                "confidence": s["confidence"],
                "is_fallback": s.get("is_fallback", False)
            }
            for s in res["signatures"]
        ],
        "document_dimensions": res["document_shape"],
        "latency_ms": res["latency_ms"]
    }
    return json.dumps(output, indent=2)

@mcp.tool()
def verify_signatures(query_signature: str, reference_signature: str) -> str:
    """
    Compares a candidate/query signature against a reference specimen signature.
    Applies pen-thickness stroke skeletonization and 256-D deep Siamese metric matching.
    Returns:
      - is_match: bool
      - verdict: VERIFIED_MATCH, PROBABLE_MATCH, INCONCLUSIVE_REVIEW, or REJECTED_FORGERY
      - similarity_percentage: float (0.0 to 100.0%)
      - l2_distance: float (Euclidean distance in 256-D embedding space)
    """
    _, ver, _ = get_engines()
    res = ver.verify_pair(query_signature, reference_signature)
    output = {
        "is_match": res["is_match"],
        "verdict": res["verdict"],
        "similarity_percentage": res["similarity_percentage"],
        "l2_distance": res["l2_distance"],
        "cosine_similarity": res["cosine_similarity"],
        "latency_ms": res["latency_ms"]
    }
    return json.dumps(output, indent=2)

@mcp.tool()
def verify_document_for_customer(
    document_path_or_base64: str,
    customer_id: str,
    document_type: str = "Document"
) -> str:
    """
    Full End-to-End Enterprise Verification:
    1. Detects signature region inside the document (cheque, NIC, passport, etc.).
    2. Fetches customer's enrolled specimen cluster from the database.
    3. Verifies detected signature against customer's enrolled signatures.
    4. Records an audit log and returns verification decision and confidence score.
    """
    det, ver, database = get_engines()
    specimens = database.get_customer_specimens(customer_id)
    if not specimens:
        return json.dumps({
            "error": f"Customer '{customer_id}' not found or has no enrolled specimens."
        }, indent=2)

    # 1. Detect
    det_res = det.detect(document_path_or_base64)
    if not det_res["detected"] or len(det_res["signatures"]) == 0:
        return json.dumps({
            "success": False,
            "verdict": "NO_SIGNATURE_DETECTED",
            "message": "No signature could be located in the submitted document."
        }, indent=2)

    query_crop = det_res["signatures"][0]["crop_image"]
    ref_embeddings = [s["embedding"] for s in specimens]

    # 2. Verify
    ver_res = ver.verify_against_specimens(query_crop, ref_embeddings)

    # 3. Audit
    database.log_verification(
        customer_id=customer_id,
        document_type=document_type,
        verdict=ver_res["verdict"],
        is_match=ver_res["is_match"],
        similarity_percentage=ver_res["similarity_percentage"],
        distance=ver_res["min_distance"]
    )

    output = {
        "customer_id": customer_id,
        "document_type": document_type,
        "signature_detected": True,
        "is_match": ver_res["is_match"],
        "verdict": ver_res["verdict"],
        "similarity_percentage": ver_res["similarity_percentage"],
        "min_embedding_distance": ver_res["min_distance"],
        "mean_cluster_distance": ver_res["mean_distance"],
        "specimens_compared": len(specimens),
        "matched_specimen_index": ver_res["matched_specimen_index"],
        "latency_ms": ver_res["latency_ms"]
    }
    return json.dumps(output, indent=2)

@mcp.tool()
def enroll_customer_signature(
    customer_id: str,
    full_name: str,
    signature_image: str,
    account_number: str = "",
    document_type: str = "NIC/Passport"
) -> str:
    """
    Enrolls a reference specimen signature for a customer into the database.
    Extracts and indexes the 256-D neural embedding vector.
    """
    _, ver, database = get_engines()
    database.enroll_customer(customer_id, full_name, account_number, document_type)
    
    # Extract embedding
    emb, _, _ = ver.extract_embedding(signature_image)
    
    # Encode for storage
    from core.preprocessor import decode_image, encode_image_base64
    bgr = decode_image(signature_image)
    b64 = encode_image_base64(bgr, format_ext=".png")
    
    specimen_idx = database.add_specimen(customer_id, b64, emb)
    return json.dumps({
        "success": True,
        "customer_id": customer_id,
        "specimen_index": specimen_idx,
        "message": f"Specimen #{specimen_idx} successfully enrolled for {full_name} ({customer_id})."
    }, indent=2)

@mcp.tool()
def list_enrolled_customers() -> str:
    """
    Lists all customers enrolled in the signature verification database
    along with their registered specimen counts.
    """
    _, _, database = get_engines()
    customers = database.list_customers()
    return json.dumps({"customers": customers}, indent=2)

if __name__ == "__main__":
    logger.info("Starting Signature Verification Agent MCP Server...")
    mcp.run()
