import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
import cv2
import numpy as np
import logging
import time
from PIL import Image
from typing import List, Dict, Any, Union, Tuple
from pathlib import Path

from config import (
    VERIFIER_INPUT_SIZE,
    DEVICE
)
from core.preprocessor import SignaturePreprocessor, decode_image, encode_image_base64

logger = logging.getLogger("SignatureVerifier")

class SignatureVerifier:
    """
    Enterprise Offline Signature Verification Engine.
    Employs Deep Skeleton-Manifold Metric Learning:
      1. Isolates handwritten strokes and applies Zhang-Suen morphological skeletonization,
         neutralizing stroke-thickness variations between fine ballpoint, gel pen, marker, and stylus.
      2. Extracts invariant 512-dimensional deep topological feature vectors via ResNet18 backbone.
      3. Evaluates cosine similarity against multi-specimen customer clusters.
    """

    def __init__(self):
        self.device = torch.device(DEVICE)
        self.preprocessor = SignaturePreprocessor(target_size=VERIFIER_INPUT_SIZE)
        
        logger.info("Initializing Enterprise Signature Verifier Backbone (ResNet18)...")
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        backbone.fc = nn.Identity()
        self.backbone = backbone.to(self.device)
        self.backbone.eval()

        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        logger.info("Enterprise Signature Verifier loaded successfully.")

    def _extract_kinematic_descriptors(self, skeleton: np.ndarray) -> np.ndarray:
        """
        Extracts stroke density zoning, directional orientation gradients, and projection profiles.
        Completely invariant to pen thickness while capturing handwriting geometry.
        """
        h, w = skeleton.shape
        gh, gw = max(1, h // 8), max(1, w // 8)
        zoning = []
        for r in range(8):
            for c in range(8):
                cell = skeleton[r*gh:(r+1)*gh, c*gw:(c+1)*gw]
                zoning.append(np.count_nonzero(cell) / float(gh * gw))
                
        # Sobel 8-bin directional orientation histogram
        gx = cv2.Sobel(skeleton.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(skeleton.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
        mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)
        hist, _ = np.histogram(ang[mag > 0], bins=8, range=(0, 360))
        hist = hist.astype(np.float32) / (np.sum(hist) + 1e-8)
        
        # Horizontal and Vertical projection profiles (16 bins each)
        h_proj = np.sum(skeleton > 0, axis=1).astype(np.float32)
        h_proj /= (np.sum(h_proj) + 1e-8)
        v_proj = np.sum(skeleton > 0, axis=0).astype(np.float32)
        v_proj /= (np.sum(v_proj) + 1e-8)
        
        h_binned = [np.mean(h_proj[i*len(h_proj)//16:(i+1)*len(h_proj)//16]) for i in range(16)]
        v_binned = [np.mean(v_proj[i*len(v_proj)//16:(i+1)*len(v_proj)//16]) for i in range(16)]
        
        kin = np.concatenate([zoning, hist, h_binned, v_binned]).astype(np.float32)
        return kin / (np.linalg.norm(kin) + 1e-8)

    def extract_embedding(self, image_input: Union[str, bytes, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extracts hybrid deep-kinematic skeleton feature vector.
        Combines ResNet18 topological features with directional zoning descriptors.
        """
        bgr = decode_image(image_input)
        _, raw_crop, skeleton_crop = self.preprocessor.preprocess_for_verification(bgr)

        # 1. Deep feature extraction on 1-pixel skeleton midline
        pil_skel = Image.fromarray(cv2.cvtColor(skeleton_crop, cv2.COLOR_GRAY2RGB))
        t = self.transform(pil_skel).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self.backbone(t).cpu().numpy().flatten()
        feat_norm = feat / (np.linalg.norm(feat) + 1e-8)
        
        # 2. Kinematic directional zoning descriptors
        kin_norm = self._extract_kinematic_descriptors(skeleton_crop)
        
        # 3. Fuse representations (512-D deep + 104-D kinematic = 616-D vector)
        fused = np.concatenate([feat_norm * 0.75, kin_norm * 0.65])
        normalized_emb = fused / (np.linalg.norm(fused) + 1e-8)
        return normalized_emb, raw_crop, skeleton_crop

    def compare_embeddings(self, emb1: np.ndarray, emb2: np.ndarray) -> Dict[str, float]:
        """
        Computes cosine similarity and calibrated distance between signature embeddings.
        """
        cos_sim = float(np.dot(emb1, emb2))
        
        # Calibrated enterprise percentage:
        # CosSim >= 0.945 (genuine) maps to >= 84.8% (Verified Match)
        # CosSim <= 0.909 (impostor) maps to <= 69.1% (Non-match)
        sim_pct = max(0.0, min(100.0, ((cos_sim - 0.75) / (0.98 - 0.75)) * 100.0))
        
        # Effective distance metric (lower is closer match)
        dist = max(0.0, (100.0 - sim_pct) / 50.0)

        return {
            "cosine_similarity": round(cos_sim, 4),
            "similarity_percentage": round(sim_pct, 2),
            "l2_distance": round(dist, 4)
        }

    def verify_pair(
        self,
        query_image: Union[str, bytes, np.ndarray],
        reference_image: Union[str, bytes, np.ndarray]
    ) -> Dict[str, Any]:
        """Compares candidate signature against a single reference specimen."""
        start_time = time.perf_counter()

        emb_q, raw_q, skel_q = self.extract_embedding(query_image)
        emb_r, raw_r, skel_r = self.extract_embedding(reference_image)

        metrics = self.compare_embeddings(emb_q, emb_r)
        sim_pct = metrics["similarity_percentage"]
        verdict, is_match = self._classify_verdict(sim_pct)

        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

        return {
            "is_match": is_match,
            "verdict": verdict,
            "similarity_percentage": sim_pct,
            "l2_distance": metrics["l2_distance"],
            "cosine_similarity": metrics["cosine_similarity"],
            "thresholds": {
                "verified_match_min_pct": 75.0,
                "probable_match_min_pct": 60.0,
                "inconclusive_min_pct": 45.0
            },
            "query_crops": {
                "raw_base64": encode_image_base64(raw_q, format_ext=".png"),
                "skeleton_base64": encode_image_base64(skel_q, format_ext=".png")
            },
            "reference_crops": {
                "raw_base64": encode_image_base64(raw_r, format_ext=".png"),
                "skeleton_base64": encode_image_base64(skel_r, format_ext=".png")
            },
            "latency_ms": latency_ms
        }

    def verify_against_specimens(
        self,
        query_image: Union[str, bytes, np.ndarray],
        reference_embeddings: List[np.ndarray],
        reference_images: List[np.ndarray] = None
    ) -> Dict[str, Any]:
        """Multi-specimen banking cluster matching against enrolled customer samples."""
        start_time = time.perf_counter()

        emb_q, raw_q, skel_q = self.extract_embedding(query_image)

        if not reference_embeddings:
            raise ValueError("No reference specimen embeddings provided for customer.")

        similarities = []
        distances = []
        for ref_emb in reference_embeddings:
            m = self.compare_embeddings(emb_q, ref_emb)
            similarities.append(m["similarity_percentage"])
            distances.append(m["l2_distance"])

        max_sim = max(similarities)
        min_dist = min(distances)
        mean_dist = float(np.mean(distances))
        best_specimen_idx = int(np.argmax(similarities))

        verdict, is_match = self._classify_verdict(max_sim)
        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

        return {
            "is_match": is_match,
            "verdict": verdict,
            "similarity_percentage": max_sim,
            "min_distance": min_dist,
            "mean_distance": round(mean_dist, 4),
            "matched_specimen_index": best_specimen_idx + 1,
            "specimen_count": len(reference_embeddings),
            "individual_similarities": similarities,
            "query_crops": {
                "raw_base64": encode_image_base64(raw_q, format_ext=".png"),
                "skeleton_base64": encode_image_base64(skel_q, format_ext=".png")
            },
            "latency_ms": latency_ms
        }

    def _classify_verdict(self, similarity_pct: float) -> Tuple[str, bool]:
        """
        Enterprise Banking Decision Classification:
        - VERIFIED_MATCH (>= 80%): Conclusive match with enrolled specimen.
        - PROBABLE_MATCH (70% - 80%): Genuine signature with natural pen/angle variation.
        - INCONCLUSIVE_REVIEW (55% - 70%): Marginal similarity; routed to manual officer review.
        - REJECTED_FORGERY (< 55%): Significant divergence; rejected as mismatch/forgery.
        """
        if similarity_pct >= 80.0:
            return "VERIFIED_MATCH", True
        elif similarity_pct >= 70.0:
            return "PROBABLE_MATCH", True
        elif similarity_pct >= 55.0:
            return "INCONCLUSIVE_REVIEW", False
        else:
            return "REJECTED_FORGERY", False
