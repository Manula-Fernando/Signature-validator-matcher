import requests
import json
from pathlib import Path
from typing import Union, List, Dict, Any, Optional

from core.preprocessor import decode_image, encode_image_base64

class SignatureAgentClient:
    """
    Enterprise Plug-in Client SDK for Signature Detection & Verification.
    Can operate in:
      - 'remote' mode: calls FastAPI REST endpoint (default: http://localhost:8000)
      - 'direct' mode: in-process direct model execution for maximum zero-latency throughput.
    """

    def __init__(self, mode: str = "direct", base_url: str = "http://localhost:8000"):
        self.mode = mode.lower()
        self.base_url = base_url.rstrip("/")

        if self.mode == "direct":
            from core.detector import SignatureDetector
            from core.verifier import SignatureVerifier
            from core.database import SpecimenDatabase
            self.detector = SignatureDetector()
            self.verifier = SignatureVerifier()
            self.db = SpecimenDatabase()

    def _to_base64(self, image_input: Any) -> str:
        bgr = decode_image(image_input)
        return encode_image_base64(bgr, format_ext=".png")

    def detect(self, document_image: Any, conf_threshold: float = 0.35) -> Dict[str, Any]:
        """Detects signature bounding boxes in document."""
        if self.mode == "direct":
            return self.detector.detect(document_image, conf_threshold=conf_threshold)

        b64 = self._to_base64(document_image)
        resp = requests.post(
            f"{self.base_url}/api/v1/detect",
            json={"image": b64, "conf_threshold": conf_threshold}
        )
        resp.raise_for_status()
        return resp.json()

    def verify_pair(self, query_signature: Any, reference_signature: Any) -> Dict[str, Any]:
        """Compares candidate signature with reference signature."""
        if self.mode == "direct":
            return self.verifier.verify_pair(query_signature, reference_signature)

        q_b64 = self._to_base64(query_signature)
        r_b64 = self._to_base64(reference_signature)
        resp = requests.post(
            f"{self.base_url}/api/v1/verify",
            json={"query_image": q_b64, "reference_image": r_b64}
        )
        resp.raise_for_status()
        return resp.json()

    def verify_document_for_customer(
        self,
        document_image: Any,
        customer_id: str,
        document_type: str = "Cheque"
    ) -> Dict[str, Any]:
        """Full end-to-end document detection + customer specimen verification."""
        if self.mode == "direct":
            specimens = self.db.get_customer_specimens(customer_id)
            if not specimens:
                return {"success": False, "error": f"Customer '{customer_id}' has no enrolled specimens."}
            det = self.detector.detect(document_image)
            if not det["detected"]:
                return {"success": True, "detected": False, "is_match": False, "verdict": "NO_SIGNATURE_DETECTED"}
            query_crop = det["signatures"][0]["crop_image"]
            ref_embs = [s["embedding"] for s in specimens]
            res = self.verifier.verify_against_specimens(query_crop, ref_embs)
            self.db.log_verification(customer_id, document_type, res["verdict"], res["is_match"], res["similarity_percentage"], res["min_distance"])
            return {
                "success": True,
                "customer_id": customer_id,
                "detected": True,
                "is_match": res["is_match"],
                "verdict": res["verdict"],
                "similarity_percentage": res["similarity_percentage"],
                "min_distance": res["min_distance"],
                "mean_distance": res["mean_distance"],
                "latency_ms": res.get("latency_ms", 0.0)
            }

        doc_b64 = self._to_base64(document_image)
        resp = requests.post(
            f"{self.base_url}/api/v1/verify-document",
            json={
                "document_image": doc_b64,
                "customer_id": customer_id,
                "document_type": document_type
            }
        )
        resp.raise_for_status()
        return resp.json()

    def enroll_customer(
        self,
        customer_id: str,
        full_name: str,
        specimen_images: List[Any],
        account_number: str = "",
        document_type: str = "NIC"
    ) -> Dict[str, Any]:
        """Enrolls customer specimen signatures."""
        b64_list = [self._to_base64(img) for img in specimen_images]
        if self.mode == "direct":
            self.db.enroll_customer(customer_id, full_name, account_number, document_type)
            count = 0
            for b64 in b64_list:
                emb, _, _ = self.verifier.extract_embedding(b64)
                self.db.add_specimen(customer_id, b64, emb)
                count += 1
            return {"success": True, "customer_id": customer_id, "specimens_enrolled": count}

        resp = requests.post(
            f"{self.base_url}/api/v1/enroll",
            json={
                "customer_id": customer_id,
                "full_name": full_name,
                "account_number": account_number,
                "document_type": document_type,
                "specimen_images": b64_list
            }
        )
        resp.raise_for_status()
        return resp.json()

    def list_customers(self) -> List[Dict[str, Any]]:
        """Lists enrolled customers."""
        if self.mode == "direct":
            return self.db.list_customers()
        resp = requests.get(f"{self.base_url}/api/v1/customers")
        resp.raise_for_status()
        return resp.json().get("customers", [])
