# Contributing to Signature-validator-matcher

Thank you for your interest in contributing to **Signature-validator-matcher (SignaVerify)**! We welcome contributions from developers, researchers, and engineers around the world. Whether you want to fix a bug, add support for new document types, improve handwriting stroke thinning algorithms, or optimize CPU inference, your help is appreciated.

---

## 🌟 Show Your Support
If you find this project useful, please consider giving it a **Star ⭐** on GitHub! It helps more developers and financial institutions discover this open-source tool.

---

## 🚀 How to Contribute

### 1. Fork & Clone
1. Fork the repository on GitHub by clicking the **Fork** button at the top right of [Signature-validator-matcher](https://github.com/your-username/Signature-validator-matcher).
2. Clone your fork locally:
   ```bash
   git clone https://github.com/<your-username>/Signature-validator-matcher.git
   cd Signature-validator-matcher
   ```

### 2. Set Up Environment
Create a virtual environment and install the dependencies:
```bash
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Create a Feature Branch
Always create a new branch from `main`:
```bash
git checkout -b feature/your-feature-name
# Or for bug fixes:
git checkout -b fix/issue-description
```

### 4. Making Changes & Testing
- Keep dependencies lightweight and offline-friendly (standard CPU-compatible).
- Preserve existing docstrings, type annotations, and error handling.
- Run the full benchmark validation suite to verify 0 regressions:
  ```bash
  python tests/comprehensive_validation.py
  ```
- Make sure all 23 pairwise tests and 8 document types pass with 100% accuracy.

### 5. Commit & Push
Write clear, conventional commit messages:
```bash
git add .
git commit -m "feat: Add support for passport signature localization"
git push origin feature/your-feature-name
```

### 6. Submit a Pull Request
1. Open a Pull Request (PR) against the `main` branch of `Signature-validator-matcher`.
2. Provide a clear description of the problem solved or feature added.
3. Include benchmark latency or accuracy test results if modifying `core/detector.py` or `core/verifier.py`.

---

## 💡 Areas Where We Welcome Contributions
- **New National Document Formats**: Templates for checks, identity cards, driving licenses, and deeds from other regions (India, UK, US, EU, Southeast Asia).
- **Novel Skeleton Thinning & Vector Extraction**: Advancements in morphological graphs, B-spline curvature vectors, or lightweight transformer backbones.
- **Client SDKs**: Go, Java, C#, or Node.js client bindings for the REST API and MCP server.
- **Docker / Containerization**: Multi-stage lightweight CPU Dockerfiles and Kubernetes deployment manifests.

---

## 📜 Code of Conduct
Please be respectful and constructive in all discussions, issues, and code reviews. We are committed to providing a welcoming, inclusive, and collaborative environment.

## 📄 License
By contributing to **Signature-validator-matcher**, you agree that your contributions will be licensed under the [MIT License](LICENSE).
