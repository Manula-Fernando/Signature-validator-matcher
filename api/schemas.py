from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class DetectRequest(BaseModel):
    image: str = Field(..., description="Base64 encoded document image (Cheque, NIC, Passport, Contract)")
    conf_threshold: Optional[float] = Field(0.35, description="Confidence threshold for YOLO signature detector")

class SignatureDetectionItem(BaseModel):
    id: int
    bbox: List[int]
    confidence: float
    crop_base64: str
    is_fallback: Optional[bool] = False

class DetectResponse(BaseModel):
    success: bool = True
    detected: bool
    count: int
    signatures: List[SignatureDetectionItem]
    annotated_image_base64: str
    latency_ms: float
    document_shape: List[int]

class VerifyPairRequest(BaseModel):
    query_image: str = Field(..., description="Base64 or path of the candidate signature to test")
    reference_image: str = Field(..., description="Base64 or path of the reference specimen signature")

class VerifyPairResponse(BaseModel):
    success: bool = True
    is_match: bool
    verdict: str = Field(..., description="VERIFIED_MATCH | PROBABLE_MATCH | INCONCLUSIVE_REVIEW | REJECTED_FORGERY")
    similarity_percentage: float
    l2_distance: float
    cosine_similarity: float
    thresholds: Dict[str, float]
    query_crops: Dict[str, str]
    reference_crops: Dict[str, str]
    latency_ms: float

class EnrollCustomerRequest(BaseModel):
    customer_id: str = Field(..., description="Customer ID, NIC, or Passport Number (e.g., CUST_1001)")
    full_name: str = Field(..., description="Customer's full name")
    account_number: Optional[str] = Field("", description="Bank account number")
    document_type: Optional[str] = Field("NIC/Passport", description="Document type")
    specimen_images: List[str] = Field(..., description="List of base64-encoded specimen signatures (1 to 5)")

class EnrollCustomerResponse(BaseModel):
    success: bool = True
    customer_id: str
    specimens_enrolled: int
    message: str

class VerifyDocumentRequest(BaseModel):
    document_image: str = Field(..., description="Base64 encoded full document (Cheque, ID, Contract)")
    customer_id: str = Field(..., description="Customer ID whose enrolled specimen signatures to compare against")
    document_type: Optional[str] = Field("Document", description="Cheque, National ID, Passport, Agreement")

class VerifyDocumentResponse(BaseModel):
    success: bool = True
    document_type: str
    customer_id: str
    customer_name: str
    detected: bool
    detection_count: int
    annotated_document_base64: str
    is_match: bool
    verdict: str
    similarity_percentage: float
    min_distance: float
    mean_distance: float
    matched_specimen_index: int
    specimens_compared: int
    query_crops: Dict[str, str]
    latency_ms: float

class CustomerItem(BaseModel):
    customer_id: str
    full_name: str
    account_number: Optional[str] = ""
    document_type: Optional[str] = ""
    specimen_count: int
    created_at: str

class HealthResponse(BaseModel):
    status: str
    version: str = "1.0.0"
    device: str
    detector_model: str
    verifier_model: str
    enrolled_customers: int
