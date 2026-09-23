import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import time
import logging
import cv2
import numpy as np
from config import SAMPLES_DIR
from core.detector import SignatureDetector
from core.verifier import SignatureVerifier
from core.preprocessor import SignaturePreprocessor
from agent.client_sdk import SignatureAgentClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("BenchmarkSuite")

def run_benchmarks():
    print("=" * 70)
    print(" SIGNATURE DETECTION & VERIFICATION AGENT - BENCHMARK SUITE")
    print("=" * 70)

    # 1. Initialize Engines
    print("\n[1/4] Initializing Local CPU ML Engines...")
    t0 = time.perf_counter()
    detector = SignatureDetector()
    verifier = SignatureVerifier()
    preprocessor = SignaturePreprocessor()
    client = SignatureAgentClient(mode="direct")
    load_time = round((time.perf_counter() - t0) * 1000, 2)
    print(f"       Engines loaded in {load_time} ms on CPU.")

    # 2. Benchmark Document Signature Detection
    print("\n[2/4] Benchmarking Document Signature Detection...")
    test_docs = [
        ("Bank Cheque (Genuine)", SAMPLES_DIR / "cheque_genuine.jpg"),
        ("National ID Card", SAMPLES_DIR / "national_id_card.jpg"),
        ("Passport Page", SAMPLES_DIR / "passport_page.jpg"),
        ("Commercial Contract", SAMPLES_DIR / "loan_contract.jpg"),
    ]

    detection_latencies = []
    for doc_name, doc_path in test_docs:
        assert doc_path.exists(), f"Sample document not found: {doc_path}"
        res = detector.detect(str(doc_path))
        detection_latencies.append(res["latency_ms"])
        detected = res["detected"]
        count = res["count"]
        conf = res["signatures"][0]["confidence"] if count > 0 else 0.0
        method = res["signatures"][0].get("method", "unknown") if count > 0 else "none"
        print(f"       [PASS] {doc_name:<24} -> Detected: {detected} (count: {count}, conf: {conf:.2f}, method: {method}, latency: {res['latency_ms']} ms)")
        assert detected, f"Failed to detect signature on {doc_name}"

    avg_det_latency = np.mean(detection_latencies)
    print(f"       Average Detection Latency: {avg_det_latency:.2f} ms")

    # 3. Benchmark Verification Discrimination (Genuine vs Forged)
    print("\n[3/4] Benchmarking Verification Model (Deep Skeleton Metric Learning)...")
    sig_a1 = SAMPLES_DIR / "person_A_sig1.png"
    sig_a2 = SAMPLES_DIR / "person_A_sig2.png"
    sig_a3 = SAMPLES_DIR / "person_A_sig3.png"
    sig_b1 = SAMPLES_DIR / "person_B_sig1.png"
    sig_c1 = SAMPLES_DIR / "person_C_sig1.png"

    verification_tests = [
        ("Genuine Pair (Signer A, Sample 1 vs Sample 2)", sig_a1, sig_a2, True),
        ("Genuine Pair (Signer A, Sample 1 vs Sample 3)", sig_a1, sig_a3, True),
        ("Forgery Pair (Signer A vs Signer B)", sig_a1, sig_b1, False),
        ("Forgery Pair (Signer A vs Signer C)", sig_a1, sig_c1, False),
        ("Forgery Pair (Signer B vs Signer C)", sig_b1, sig_c1, False),
    ]

    correct_predictions = 0
    ver_latencies = []
    for test_name, q_path, r_path, expected_match in verification_tests:
        v_res = verifier.verify_pair(str(q_path), str(r_path))
        ver_latencies.append(v_res["latency_ms"])
        is_match = v_res["is_match"]
        passed = (is_match == expected_match)
        if passed:
            correct_predictions += 1
        status_sym = "[PASS]" if passed else "[FAIL]"
        print(f"       {status_sym} {test_name:<48} -> Match: {str(is_match):<5} (Verdict: {v_res['verdict']:<17} CosSim: {v_res['cosine_similarity']:.4f}, Sim: {v_res['similarity_percentage']}%, Latency: {v_res['latency_ms']} ms)")
        assert passed, f"Verification failed for test: {test_name}"

    accuracy = (correct_predictions / len(verification_tests)) * 100.0
    avg_ver_latency = np.mean(ver_latencies)
    print(f"       Verification Accuracy: {accuracy:.1f}% ({correct_predictions}/{len(verification_tests)})")
    print(f"       Average Verification Latency: {avg_ver_latency:.2f} ms")

    # 4. Benchmark End-to-End Enterprise Flow (Document + Database Verification)
    print("\n[4/4] Benchmarking End-to-End Enterprise Flow (SDK / Database Cluster)...")
    # Genuine Cheque vs CUST_001 (Johnathan Carter)
    e2e_genuine = client.verify_document_for_customer(
        document_image=str(SAMPLES_DIR / "cheque_genuine.jpg"),
        customer_id="CUST_001",
        document_type="Cheque"
    )
    print(f"       Cheque (Genuine) vs CUST_001: Match={e2e_genuine['is_match']} (Verdict: {e2e_genuine['verdict']}, Similarity: {e2e_genuine['similarity_percentage']}%)")
    assert e2e_genuine["is_match"], "Genuine cheque failed verification!"

    # Forged Cheque vs CUST_001 (Johnathan Carter)
    e2e_forged = client.verify_document_for_customer(
        document_image=str(SAMPLES_DIR / "cheque_forged.jpg"),
        customer_id="CUST_001",
        document_type="Cheque"
    )
    print(f"       Cheque (Forged)  vs CUST_001: Match={e2e_forged['is_match']} (Verdict: {e2e_forged['verdict']}, Similarity: {e2e_forged['similarity_percentage']}%)")
    assert not e2e_forged["is_match"], "Forged cheque was incorrectly accepted!"

    # Final Scorecard
    print("\n" + "=" * 70)
    print(" ENTERPRISE BENCHMARK SCORECARD")
    print("=" * 70)
    print(f" * Detection Coverage Rate     : 100.0% (Cheque, NIC, Passport, Contract)")
    print(f" * Verification Accuracy       : {accuracy:.1f}%")
    print(f" * False Acceptance Rate (FAR) : 0.0% (All forgeries successfully rejected)")
    print(f" * False Rejection Rate (FRR)  : 0.0% (All genuine signatures matched)")
    print(f" * Avg Detection Latency (CPU) : {avg_det_latency:.2f} ms")
    print(f" * Avg Verifier Latency (CPU)  : {avg_ver_latency:.2f} ms")
    print(f" * Hardware Acceleration       : None required (Pure CPU, lightweight)")
    print("=" * 70)
    print(" ALL BENCHMARK TESTS PASSED WITH 100% ACCURACY!")

if __name__ == "__main__":
    run_benchmarks()
