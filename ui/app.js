// SignaVerify Frontend Controller
const API_BASE = "";

// State variables
let currentDocumentBase64 = null;
let currentPresetId = "cheque_genuine";
let sampleCatalog = null;
let pairSig1Base64 = null;
let pairSig2Base64 = null;

// Initialize on DOM ready
document.addEventListener("DOMContentLoaded", () => {
  initTabs();
  loadSystemStatus();
  loadSampleCatalog();
  loadCustomerRegistry();
  loadAuditLogs();
  setupDocumentDropzone();
  setupPairwiseDropzones();
  setupEnrollmentForm();

  // Action Buttons
  document.getElementById("btn-verify-doc").addEventListener("click", executeDocumentVerification);
  document.getElementById("btn-verify-pair").addEventListener("click", executePairwiseVerification);
});

// ----------------- TAB SWITCHING ----------------- //
function initTabs() {
  const tabs = document.querySelectorAll(".tab-btn");
  tabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      tabs.forEach((t) => t.classList.remove("active"));
      document.querySelectorAll(".tab-content").forEach((c) => c.classList.remove("active"));

      tab.classList.add("active");
      const targetId = tab.getAttribute("data-tab");
      const targetContent = document.getElementById(targetId);
      if (targetContent) targetContent.classList.add("active");

      // Auto-refresh data on tab change
      if (targetId === "tab-customers") loadCustomerRegistry();
      if (targetId === "tab-audits") loadAuditLogs();
    });
  });
}

// ----------------- SYSTEM HEALTH ----------------- //
async function loadSystemStatus() {
  try {
    const res = await fetch(`${API_BASE}/api/v1/health`);
    if (res.ok) {
      const data = await res.json();
      document.getElementById("status-device").textContent = data.device.toUpperCase();
    }
  } catch (err) {
    console.warn("Could not fetch health status:", err);
  }
}

// ----------------- SAMPLE CATALOG & PRESETS ----------------- //
async function loadSampleCatalog() {
  try {
    const res = await fetch(`${API_BASE}/api/v1/samples`);
    if (!res.ok) return;
    sampleCatalog = await res.json();

    // Preset button click handlers
    const chips = document.querySelectorAll(".preset-chip");
    chips.forEach((chip) => {
      chip.addEventListener("click", () => {
        chips.forEach((c) => c.classList.remove("active"));
        chip.classList.add("active");
        const presetId = chip.getAttribute("data-preset");
        loadPresetDocument(presetId);
      });
    });

    // Load initial default preset
    loadPresetDocument("cheque_genuine");
  } catch (err) {
    console.warn("Could not load sample catalog:", err);
  }
}

async function loadPresetDocument(presetId) {
  currentPresetId = presetId;
  const docMeta = sampleCatalog?.documents?.find((d) => d.id === presetId);
  if (!docMeta) return;

  // Auto-set customer dropdown if specified
  if (docMeta.expected_customer_id) {
    const custSelect = document.getElementById("customer-select");
    custSelect.value = docMeta.expected_customer_id;
  }

  // Load image from backend sample directory via relative fetch
  try {
    const imgUrl = `/dataset/samples/${docMeta.filename}`;
    const imgResponse = await fetch(imgUrl);
    const blob = await imgResponse.blob();
    const reader = new FileReader();
    reader.onloadend = () => {
      currentDocumentBase64 = reader.result;
      document.getElementById("doc-preview-img").src = currentDocumentBase64;
      resetVerdictDisplay(docMeta.title);
    };
    reader.readAsDataURL(blob);
  } catch (err) {
    console.error("Error loading preset image:", err);
  }
}

function resetVerdictDisplay(title) {
  const pill = document.getElementById("verdict-pill");
  pill.className = "verdict-badge pending";
  pill.textContent = "Awaiting Verification";

  document.getElementById("detection-count-tag").textContent = "Ready";
  document.getElementById("metric-similarity").textContent = "0.0%";
  document.getElementById("metric-gauge-fill").style.width = "0%";
  document.getElementById("metric-dist").textContent = "0.0000";
  document.getElementById("metric-det-lat").textContent = "0.0 ms";
  document.getElementById("metric-ver-lat").textContent = "0.0 ms";

  document.getElementById("crop-raw").src = "";
  document.getElementById("crop-skeleton").src = "";
  document.getElementById("crop-specimen").src = "";

  document.getElementById("explanation-text").textContent =
    `Loaded preset "${title}". Click "Execute End-to-End Verification" to run the YOLO locator and Siamese matching engine.`;
}

// ----------------- DOCUMENT UPLOAD DROPZONE ----------------- //
function setupDocumentDropzone() {
  const dropzone = document.getElementById("doc-dropzone");
  const fileInput = document.getElementById("doc-file-input");

  dropzone.addEventListener("click", () => fileInput.click());

  dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.style.borderColor = "#6366f1";
  });

  dropzone.addEventListener("dragleave", () => {
    dropzone.style.borderColor = "";
  });

  dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.style.borderColor = "";
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleUploadedDocument(e.dataTransfer.files[0]);
    }
  });

  fileInput.addEventListener("change", (e) => {
    if (e.target.files && e.target.files[0]) {
      handleUploadedDocument(e.target.files[0]);
    }
  });
}

function handleUploadedDocument(file) {
  const reader = new FileReader();
  reader.onloadend = () => {
    currentDocumentBase64 = reader.result;
    document.getElementById("doc-preview-img").src = currentDocumentBase64;
    document.querySelectorAll(".preset-chip").forEach((c) => c.classList.remove("active"));
    resetVerdictDisplay(file.name);
  };
  reader.readAsDataURL(file);
}

// ----------------- EXECUTE E2E VERIFICATION ----------------- //
async function executeDocumentVerification() {
  if (!currentDocumentBase64) {
    alert("Please select or upload a document first.");
    return;
  }

  const customerId = document.getElementById("customer-select").value;
  const docType = document.getElementById("doc-type-select").value;
  const btn = document.getElementById("btn-verify-doc");
  btn.disabled = true;
  btn.innerHTML = `<span class="spinner"></span> Processing AI Inference...`;

  try {
    const response = await fetch(`${API_BASE}/api/v1/verify-document`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        document_image: currentDocumentBase64,
        customer_id: customerId,
        document_type: docType
      })
    });

    if (!response.ok) {
      const err = await response.json();
      alert(`Verification failed: ${err.detail || "Server error"}`);
      return;
    }

    const data = await response.json();
    renderVerificationResults(data);
  } catch (err) {
    console.error("Verification request error:", err);
    alert("Failed to connect to verification API.");
  } finally {
    btn.disabled = false;
    btn.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg> Execute End-to-End Verification`;
  }
}

function renderVerificationResults(data) {
  const pill = document.getElementById("verdict-pill");
  const countTag = document.getElementById("detection-count-tag");
  const simText = document.getElementById("metric-similarity");
  const gaugeFill = document.getElementById("metric-gauge-fill");
  const distText = document.getElementById("metric-dist");
  const detLatText = document.getElementById("metric-det-lat");
  const verLatText = document.getElementById("metric-ver-lat");
  const explText = document.getElementById("explanation-text");

  // Annotated Document
  if (data.annotated_document_base64) {
    document.getElementById("doc-preview-img").src = data.annotated_document_base64;
  }

  countTag.textContent = `${data.detection_count} Signature(s) Located`;

  // Verdict Pill
  pill.className = "verdict-badge";
  if (data.verdict === "VERIFIED_MATCH") {
    pill.classList.add("match");
    pill.textContent = "VERIFIED MATCH (HIGH CONFIDENCE)";
    explText.innerHTML = `<strong>AUTHENTIC SIGNATURE CONFIRMED</strong>: The detected stroke topology on this ${data.document_type} matched specimen #${data.matched_specimen_index} on file for <strong>${data.customer_name}</strong> with <strong>${data.similarity_percentage}%</strong> confidence. Pen thickness variation was successfully neutralized.`;
  } else if (data.verdict === "PROBABLE_MATCH") {
    pill.classList.add("probable");
    pill.textContent = "PROBABLE MATCH (NATURAL VARIATION)";
    explText.innerHTML = `<strong>PROBABLE MATCH</strong>: The signature matched customer <strong>${data.customer_name}</strong> (${data.similarity_percentage}% similarity). Natural handwriting variations (pen pressure, surface tilt) were detected but within banking tolerance.`;
  } else if (data.verdict === "INCONCLUSIVE_REVIEW") {
    pill.classList.add("probable");
    pill.textContent = "INCONCLUSIVE (OFFICER REVIEW)";
    explText.innerHTML = `<strong>MARGINAL SIMILARITY (${data.similarity_percentage}%)</strong>: Signature displays some structural deviation. Flagged for secondary manual KYC officer verification.`;
  } else {
    pill.classList.add("forgery");
    pill.textContent = "REJECTED (SUSPECTED FORGERY / MISMATCH)";
    explText.innerHTML = `<strong>SECURITY ALERT - MISMATCH DETECTED</strong>: The signature on this ${data.document_type} diverges statistically from customer <strong>${data.customer_name}</strong>'s specimen cluster (Similarity: <strong>${data.similarity_percentage}%</strong>, Distance: ${data.min_distance}). Transaction should be rejected or escalated to fraud operations.`;
  }

  // Scorecard
  simText.textContent = `${data.similarity_percentage}%`;
  gaugeFill.style.width = `${Math.min(100, Math.max(0, data.similarity_percentage))}%`;
  if (data.similarity_percentage >= 70) {
    gaugeFill.style.background = "linear-gradient(90deg, #6366f1, #10b981)";
  } else if (data.similarity_percentage >= 50) {
    gaugeFill.style.background = "linear-gradient(90deg, #6366f1, #f59e0b)";
  } else {
    gaugeFill.style.background = "linear-gradient(90deg, #f59e0b, #ef4444)";
  }

  distText.textContent = data.min_distance.toFixed(4);
  detLatText.textContent = `${(data.latency_ms * 0.7).toFixed(1)} ms`;
  verLatText.textContent = `${(data.latency_ms * 0.3).toFixed(1)} ms`;

  // Stroke Inspector Images
  if (data.query_crops) {
    document.getElementById("crop-raw").src = data.query_crops.raw_base64 || "";
    document.getElementById("crop-skeleton").src = data.query_crops.skeleton_base64 || "";
  }

  // Load matched specimen image
  loadMatchedSpecimenPreview(data.customer_id, data.matched_specimen_index);
}

async function loadMatchedSpecimenPreview(customerId, specimenIndex) {
  try {
    const res = await fetch(`${API_BASE}/api/v1/customers/${customerId}/specimens`);
    if (res.ok) {
      const data = await res.json();
      const spec = data.specimens.find((s) => s.specimen_index === specimenIndex) || data.specimens[0];
      if (spec) {
        document.getElementById("crop-specimen").src = spec.image_base64;
        document.getElementById("specimen-label").textContent = `Enrolled Specimen #${spec.specimen_index}`;
      }
    }
  } catch (err) {
    console.warn("Could not load specimen preview:", err);
  }
}

// ----------------- PAIRWISE VERIFICATION ----------------- //
function setupPairwiseDropzones() {
  const dz1 = document.getElementById("pair-dropzone-1");
  const f1 = document.getElementById("pair-file-1");
  const dz2 = document.getElementById("pair-dropzone-2");
  const f2 = document.getElementById("pair-file-2");

  dz1.addEventListener("click", () => f1.click());
  dz2.addEventListener("click", () => f2.click());

  f1.addEventListener("change", (e) => {
    if (e.target.files && e.target.files[0]) {
      const reader = new FileReader();
      reader.onloadend = () => {
        pairSig1Base64 = reader.result;
        document.getElementById("pair-preview-1").src = pairSig1Base64;
      };
      reader.readAsDataURL(e.target.files[0]);
    }
  });

  f2.addEventListener("change", (e) => {
    if (e.target.files && e.target.files[0]) {
      const reader = new FileReader();
      reader.onloadend = () => {
        pairSig2Base64 = reader.result;
        document.getElementById("pair-preview-2").src = pairSig2Base64;
      };
      reader.readAsDataURL(e.target.files[0]);
    }
  });
}

async function executePairwiseVerification() {
  if (!pairSig1Base64 || !pairSig2Base64) {
    alert("Please upload both Candidate and Reference signature images.");
    return;
  }

  const btn = document.getElementById("btn-verify-pair");
  btn.disabled = true;
  btn.textContent = "Comparing...";

  try {
    const res = await fetch(`${API_BASE}/api/v1/verify`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query_image: pairSig1Base64,
        reference_image: pairSig2Base64
      })
    });

    if (!res.ok) {
      alert("Pairwise comparison failed.");
      return;
    }

    const data = await res.json();
    const banner = document.getElementById("pair-result-banner");
    const tag = document.getElementById("pair-verdict");
    banner.style.display = "flex";
    tag.textContent = data.verdict;
    tag.style.color = data.is_match ? "#34d399" : "#f87171";

    document.getElementById("pair-sim").textContent = `${data.similarity_percentage}%`;
    document.getElementById("pair-cos").textContent = data.cosine_similarity.toFixed(4);
    document.getElementById("pair-lat").textContent = `${data.latency_ms} ms`;
  } catch (err) {
    console.error("Pairwise error:", err);
  } finally {
    btn.disabled = false;
    btn.textContent = "Compare";
  }
}

// ----------------- CUSTOMER REGISTRY ----------------- //
async function loadCustomerRegistry() {
  try {
    const res = await fetch(`${API_BASE}/api/v1/customers`);
    if (!res.ok) return;
    const data = await res.json();

    const tbody = document.getElementById("customer-tbody");
    tbody.innerHTML = "";

    const select = document.getElementById("customer-select");
    select.innerHTML = "";

    data.customers.forEach((c) => {
      // Add row
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td><strong>${c.customer_id}</strong></td>
        <td>${c.full_name}</td>
        <td><code>${c.account_number || "N/A"}</code></td>
        <td>${c.document_type || "Passport/NIC"}</td>
        <td><span class="small-tag active">${c.specimen_count} Specimens</span></td>
      `;
      tbody.appendChild(tr);

      // Add to select
      const opt = document.createElement("option");
      opt.value = c.customer_id;
      opt.textContent = `${c.full_name} (${c.customer_id}) - ${c.specimen_count} Specimens`;
      select.appendChild(opt);
    });
  } catch (err) {
    console.warn("Could not load customers:", err);
  }
}

function setupEnrollmentForm() {
  const form = document.getElementById("enroll-form");
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const id = document.getElementById("enroll-id").value.trim();
    const name = document.getElementById("enroll-name").value.trim();
    const account = document.getElementById("enroll-account").value.trim();
    const doc = document.getElementById("enroll-doc").value.trim();
    const files = document.getElementById("enroll-files").files;

    if (!files || files.length === 0) {
      alert("Please choose at least 1 specimen image.");
      return;
    }

    // Convert files to base64
    const b64Promises = Array.from(files).map((file) => {
      return new Promise((resolve) => {
        const reader = new FileReader();
        reader.onloadend = () => resolve(reader.result);
        reader.readAsDataURL(file);
      });
    });

    const b64Images = await Promise.all(b64Promises);

    try {
      const res = await fetch(`${API_BASE}/api/v1/enroll`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          customer_id: id,
          full_name: name,
          account_number: account,
          document_type: doc,
          specimen_images: b64Images
        })
      });

      if (!res.ok) {
        alert("Enrollment failed.");
        return;
      }

      alert(`Successfully enrolled customer ${id}!`);
      form.reset();
      loadCustomerRegistry();
    } catch (err) {
      console.error("Enrollment error:", err);
    }
  });
}

// ----------------- AUDIT LOGS ----------------- //
async function loadAuditLogs() {
  try {
    const res = await fetch(`${API_BASE}/api/v1/audits?limit=25`);
    if (!res.ok) return;
    const data = await res.json();
    const tbody = document.getElementById("audit-tbody");
    tbody.innerHTML = "";

    data.audits.forEach((a) => {
      const tr = document.createElement("tr");
      const matchBadge = a.is_match
        ? `<span class="small-tag active">MATCH</span>`
        : `<span class="small-tag" style="background: rgba(239,68,68,0.2); color:#f87171">FORGERY</span>`;

      tr.innerHTML = `
        <td>#${a.id}</td>
        <td><strong>${a.customer_id}</strong></td>
        <td>${a.document_type}</td>
        <td><code>${a.verdict}</code></td>
        <td>${matchBadge}</td>
        <td><strong>${a.similarity_percentage.toFixed(1)}%</strong></td>
        <td style="color: var(--text-dim); font-size: 12px;">${a.timestamp}</td>
      `;
      tbody.appendChild(tr);
    });
  } catch (err) {
    console.warn("Could not load audits:", err);
  }
}
