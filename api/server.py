import logging
import time
from pathlib import Path
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

from config import (
    STATIC_DIR,
    SAMPLES_DIR,
    DEVICE,
    DETECTOR_MODEL_FILE,
    VERIFIER_MODEL_FILE,
    SERVER_HOST,
    SERVER_PORT
)
from core.detector import SignatureDetector
from core.verifier import SignatureVerifier
from core.database import SpecimenDatabase
from core.preprocessor import decode_image, encode_image_base64
from api.schemas import (
    DetectRequest,
    DetectResponse,
    VerifyPairRequest,
    VerifyPairResponse,
    EnrollCustomerRequest,
    EnrollCustomerResponse,
    VerifyDocumentRequest,
    VerifyDocumentResponse,
    HealthResponse
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("SignatureAgentAPI")

# Initialize FastAPI app
app = FastAPI(
    title="Signature Detection & Verification Agent API",
    description="Enterprise REST API for offline document signature localization, pen-invariant verification, and customer specimen cluster matching.",
    version="1.0.0"
)

# Enable CORS for enterprise plug-and-play
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global engine instances (lazy initialized)
detector: SignatureDetector = None
verifier: SignatureVerifier = None
db: SpecimenDatabase = None

@app.on_event("startup")
def startup_event():
    global detector, verifier, db
    logger.info("Initializing Signature Agent Core Engines...")
    db = SpecimenDatabase()
    detector = SignatureDetector()
    verifier = SignatureVerifier()
    logger.info("Signature Agent Core Engines successfully loaded.")

# ----------------- Core API Endpoints ----------------- #

@app.get("/api/v1/health", response_model=HealthResponse, tags=["System"])
def health_check():
    """Returns system status, device configuration, and loaded model details."""
    customers = db.list_customers() if db else []
    return HealthResponse(
        status="healthy",
        device=DEVICE,
        detector_model=DETECTOR_MODEL_FILE,
        verifier_model=VERIFIER_MODEL_FILE,
        enrolled_customers=len(customers)
    )

@app.post("/api/v1/detect", response_model=DetectResponse, tags=["Detection"])
def detect_signatures(req: DetectRequest):
    """
    Detects and localizes handwritten signature regions within any document
    (cheque, national ID, passport, contract, or form) using fine-tuned YOLO11s.
    """
    try:
        res = detector.detect(req.image, conf_threshold=req.conf_threshold)
        return DetectResponse(
            success=True,
            detected=res["detected"],
            count=res["count"],
            signatures=res["signatures"],
            annotated_image_base64=res["annotated_image_base64"],
            latency_ms=res["latency_ms"],
            document_shape=res["document_shape"]
        )
    except Exception as e:
        logger.error(f"Error in detect_signatures: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/verify", response_model=VerifyPairResponse, tags=["Verification"])
def verify_signature_pair(req: VerifyPairRequest):
    """
    Compares a candidate signature against a reference signature image.
    Applies stroke skeletonization (pen-thickness normalization) and
    deep Siamese metric distance evaluation.
    """
    try:
        res = verifier.verify_pair(req.query_image, req.reference_image)
        return VerifyPairResponse(
            success=True,
            is_match=res["is_match"],
            verdict=res["verdict"],
            similarity_percentage=res["similarity_percentage"],
            l2_distance=res["l2_distance"],
            cosine_similarity=res["cosine_similarity"],
            thresholds=res["thresholds"],
            query_crops=res["query_crops"],
            reference_crops=res["reference_crops"],
            latency_ms=res["latency_ms"]
        )
    except Exception as e:
        logger.error(f"Error in verify_signature_pair: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/enroll", response_model=EnrollCustomerResponse, tags=["Enrollment"])
def enroll_customer(req: EnrollCustomerRequest):
    """
    Registers a customer profile and enrolls 1 to 5 specimen signatures.
    Extracts and stores 256-D metric embeddings in the local SQLite database.
    """
    try:
        if not req.specimen_images:
            raise HTTPException(status_code=400, detail="At least one specimen image is required.")

        db.enroll_customer(
            customer_id=req.customer_id,
            full_name=req.full_name,
            account_number=req.account_number,
            document_type=req.document_type
        )

        enrolled_count = 0
        for img_b64 in req.specimen_images:
            emb, _, _ = verifier.extract_embedding(img_b64)
            db.add_specimen(req.customer_id, img_b64, emb)
            enrolled_count += 1

        return EnrollCustomerResponse(
            success=True,
            customer_id=req.customer_id,
            specimens_enrolled=enrolled_count,
            message=f"Successfully enrolled {enrolled_count} specimen(s) for customer {req.customer_id}"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in enroll_customer: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/verify-document", response_model=VerifyDocumentResponse, tags=["Pipeline"])
def verify_document(req: VerifyDocumentRequest):
    """
    End-to-End Enterprise Flow:
      1. Receives document (cheque, NIC, passport, contract) and customer ID.
      2. Automatically detects and crops the signature region via YOLO11s.
      3. Retrieves customer's enrolled specimen cluster from database.
      4. Evaluates multi-specimen distance and pen-invariant structural agreement.
      5. Records an immutable audit trail entry and returns a comprehensive verification report.
    """
    start_time = time.perf_counter()
    try:
        # Check customer enrollment
        specimens = db.get_customer_specimens(req.customer_id)
        if not specimens:
            raise HTTPException(
                status_code=404,
                detail=f"Customer '{req.customer_id}' not found or has no enrolled specimens."
            )

        # Lookup customer info
        customers = db.list_customers()
        cust_info = next((c for c in customers if c["customer_id"] == req.customer_id), None)
        cust_name = cust_info["full_name"] if cust_info else "Unknown"

        # Step 1: Detect signature in document
        det_result = detector.detect(req.document_image)
        if not det_result["detected"] or len(det_result["signatures"]) == 0:
            return VerifyDocumentResponse(
                success=True,
                document_type=req.document_type,
                customer_id=req.customer_id,
                customer_name=cust_name,
                detected=False,
                detection_count=0,
                annotated_document_base64=det_result["annotated_image_base64"],
                is_match=False,
                verdict="NO_SIGNATURE_DETECTED",
                similarity_percentage=0.0,
                min_distance=2.0,
                mean_distance=2.0,
                matched_specimen_index=0,
                specimens_compared=len(specimens),
                query_crops={"raw_base64": "", "skeleton_base64": ""},
                latency_ms=round((time.perf_counter() - start_time) * 1000, 2)
            )

        # Use primary detected signature
        primary_sig = det_result["signatures"][0]
        query_crop = primary_sig["crop_image"]

        # Step 2: Extract embeddings from enrolled customer specimens
        ref_embeddings = [s["embedding"] for s in specimens]

        # Step 3: Compare against specimen cluster
        ver_result = verifier.verify_against_specimens(
            query_image=query_crop,
            reference_embeddings=ref_embeddings
        )

        total_latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

        # Step 4: Audit log
        db.log_verification(
            customer_id=req.customer_id,
            document_type=req.document_type,
            verdict=ver_result["verdict"],
            is_match=ver_result["is_match"],
            similarity_percentage=ver_result["similarity_percentage"],
            distance=ver_result["min_distance"]
        )

        return VerifyDocumentResponse(
            success=True,
            document_type=req.document_type,
            customer_id=req.customer_id,
            customer_name=cust_name,
            detected=True,
            detection_count=det_result["count"],
            annotated_document_base64=det_result["annotated_image_base64"],
            is_match=ver_result["is_match"],
            verdict=ver_result["verdict"],
            similarity_percentage=ver_result["similarity_percentage"],
            min_distance=ver_result["min_distance"],
            mean_distance=ver_result["mean_distance"],
            matched_specimen_index=ver_result["matched_specimen_index"],
            specimens_compared=len(specimens),
            query_crops=ver_result["query_crops"],
            latency_ms=total_latency_ms
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in verify_document: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/customers", tags=["Customers"])
def get_customers():
    """Lists all enrolled customers with specimen counts."""
    return {"customers": db.list_customers()}

@app.get("/api/v1/customers/{customer_id}/specimens", tags=["Customers"])
def get_customer_specimens(customer_id: str):
    """Fetches all registered specimen images for a specific customer."""
    specimens = db.get_customer_specimens(customer_id)
    return {
        "customer_id": customer_id,
        "specimens": [
            {
                "specimen_index": s["specimen_index"],
                "image_base64": s["image_base64"],
                "created_at": s["created_at"]
            }
            for s in specimens
        ]
    }

@app.get("/api/v1/audits", tags=["Audit"])
def get_audits(limit: int = 20):
    """Retrieves recent verification audit log history."""
    return {"audits": db.get_recent_audits(limit=limit)}

@app.get("/api/v1/samples", tags=["Samples"])
def get_samples():
    """Returns sample documents and signatures available for 1-click POC demonstration."""
    import json
    index_file = SAMPLES_DIR / "index.json"
    if index_file.exists():
        with open(index_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"documents": [], "signatures": []}

# Mount sample assets for browser preview
if SAMPLES_DIR.exists():
    app.mount("/dataset/samples", StaticFiles(directory=str(SAMPLES_DIR)), name="samples")

# Mount static web UI at root
if STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.server:app", host=SERVER_HOST, port=SERVER_PORT, reload=False)
