# app.py
# OCR engine: Tesseract (100% local, free, no API key needed)
# Install once:
#   pip install pytesseract pillow pdf2image streamlit crewai
#   sudo apt-get install tesseract-ocr poppler-utils   (Linux / WSL)
#   brew install tesseract poppler                      (Mac)

import re
import io
import traceback

import streamlit as st
from PIL import Image, ImageFilter, ImageOps
import pytesseract
from crewai import Crew

from agents import document_agent, identity_agent, risk_agent, compliance_agent
from tasks import create_tasks, _parse_income

import pytesseract

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)



# ── Page config ──────────────────────────────────────────────── #
st.set_page_config(
    page_title="KYC Multi-Agent Dashboard",
    page_icon="🏦",
    layout="wide"
)
st.title("🏦 Multi-Agent KYC Verification System")
st.caption(
    "Powered by CrewAI + Ollama + Tesseract OCR  |  "
    "Research Project — Emergent Behavior in Multi-Agent Systems"
)


# ================================================================
#        LOCAL OCR ENGINE  (Tesseract — zero API cost)
# ================================================================

DOC_LABEL_MAP = {
    "aadhaar":  "Aadhaar Card",
    "pan":      "PAN Card",
    "passport": "Passport",
    "voter_id": "Voter ID",
}

# Keywords that identify each document type from raw OCR text
DOC_KEYWORDS = {
    "aadhaar": [
        "aadhaar", "aadhar", "aadhaara", "uidai",
        "unique identification", "enrolment no", "enrollment no",
        "vid ", "your aadhaar",
    ],
    "pan": [
        "income tax department", "permanent account number",
        "govt. of india", "government of india", "income tax",
    ],
    "passport": [
        "passport", "republic of india", "type p",
        "nationality indian", "place of issue",
    ],
    "voter_id": [
        "election commission", "electors photo", "epic", "voter",
        "electoral roll", "part no",
    ],
}

# Cue words printed near the holder's name on each document type
NAME_CUES = {
    "aadhaar":  ["name", "\u0928\u093e\u092e"],           # "naam" in Devanagari
    "pan":      ["name", "\u0928\u093e\u092e"],
    "passport": ["name", "surname", "given name"],
    "voter_id": ["name", "electors name", "elector's name"],
}

# Words that look capitalised but are never a person's name
_SKIP_WORDS = {
    "INDIA", "GOVERNMENT", "DEPARTMENT", "AUTHORITY", "COMMISSION",
    "ELECTION", "INCOME", "TAX", "UIDAI", "PAN", "EPIC", "DOB",
    "DATE", "BIRTH", "MALE", "FEMALE", "VOTER", "CARD", "REPUBLIC",
    "PERMANENT", "ACCOUNT", "NUMBER", "UNIQUE", "IDENTIFICATION",
}


# ── Image helpers ─────────────────────────────────────────────── #

def _preprocess(img: Image.Image) -> Image.Image:
    """
    Greyscale → auto-contrast → sharpen → upscale to ≥1000 px wide.
    Dramatically improves Tesseract accuracy on phone-camera ID photos.
    """
    img = img.convert("L")
    img = ImageOps.autocontrast(img, cutoff=1)
    img = img.filter(ImageFilter.SHARPEN)
    w, h = img.size
    if w < 1000:
        scale = 1000 / w
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return img


def _load_pages(uploaded_file) -> list:
    """
    Returns a list of PIL Images.
    Supports JPG / PNG / WEBP natively; converts PDF via pdf2image.
    """
    data = uploaded_file.read()
    uploaded_file.seek(0)

    if uploaded_file.type == "application/pdf":
        try:
            from pdf2image import convert_from_bytes
            return convert_from_bytes(data, dpi=250, first_page=1, last_page=2)
        except Exception as exc:
            raise RuntimeError(
                f"PDF → image conversion failed. "
                f"Is poppler installed?  Error: {exc}"
            )
    return [Image.open(io.BytesIO(data))]


# ── Document-type detection ───────────────────────────────────── #

def _detect_doc_type(text: str) -> tuple:
    """Return (doc_type, confidence) by counting keyword hits in OCR text."""
    lower = text.lower()
    scores = {
        dtype: sum(1 for kw in kws if kw in lower)
        for dtype, kws in DOC_KEYWORDS.items()
    }
    scores = {k: v for k, v in scores.items() if v > 0}
    if not scores:
        return "unknown", "low"
    best = max(scores, key=scores.get)
    conf = "high" if scores[best] >= 2 else "medium"
    return best, conf


# ── Name extraction ───────────────────────────────────────────── #

def _looks_like_name(text: str) -> bool:
    """True if text is 2-5 words, all alphabetic (no digits, symbols)."""
    words = text.strip().split()
    return (2 <= len(words) <= 5) and all(
        re.match(r"^[A-Za-z.\'-]+$", w) for w in words
    )


def _clean_name(text: str) -> str:
    """Strip leading/trailing non-alpha chars, collapse spaces, Title-Case."""
    text = re.sub(r"^[^A-Za-z]+|[^A-Za-z]+$", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.title()


def _extract_name(text: str, doc_type: str) -> str:
    """
    Two-pass name extraction from OCR text:

    Pass 1 — Cue-label search:
      Find a line containing a known label ("Name", "नाम", etc.).
      Grab the text after the colon on that line, or the very next line.

    Pass 2 — Capitalised-cluster scan:
      Regex hunt for 2-4 consecutive Title-Case or ALL-CAPS words
      that don't match known non-name tokens (INDIA, GOVERNMENT, …).
    """
    cues  = NAME_CUES.get(doc_type, ["name"])
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    # Pass 1
    for i, line in enumerate(lines):
        ll = line.lower()
        for cue in cues:
            if cue in ll:
                if ":" in line:
                    after = line.split(":", 1)[1].strip()
                    if after and _looks_like_name(after):
                        return _clean_name(after)
                if i + 1 < len(lines) and _looks_like_name(lines[i + 1]):
                    return _clean_name(lines[i + 1])

    # Pass 2 — capitalised cluster scan
    pattern = re.compile(
        r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\b"   # Title Case
        r"|\b([A-Z]{2,}(?:\s+[A-Z]{2,}){1,3})\b"       # ALL CAPS
    )
    best = ""
    for m in pattern.finditer(text):
        token = (m.group(1) or m.group(2) or "").strip()
        words = token.split()
        if any(w.upper() in _SKIP_WORDS for w in words):
            continue
        if 2 <= len(words) <= 4 and len(token) > len(best):
            best = token

    return _clean_name(best) if best else ""


# ── Main OCR entry point ──────────────────────────────────────── #

def extract_info_from_document(uploaded_file) -> dict:
    """
    Fully local OCR pipeline — no internet, no API key.

    1. Load image / convert PDF pages
    2. Pre-process each page
    3. Run Tesseract (LSTM engine, auto page-segmentation)
    4. Detect doc type by keyword scoring
    5. Extract holder's name by cue-label + regex fallback
    6. Return structured dict
    """
    try:
        pages = _load_pages(uploaded_file)
    except Exception as exc:
        return {
            "document_type": "unknown", "official_name": "",
            "confidence": "low", "notes": str(exc), "raw_text": ""
        }

    full_text  = ""
    ocr_errors = []
    for page_img in pages:
        try:
            proc      = _preprocess(page_img)
            page_text = pytesseract.image_to_string(
                proc, config="--oem 3 --psm 3 -l eng"
            )
            full_text += page_text + "\n"
        except Exception as exc:
            ocr_errors.append(str(exc))

    if not full_text.strip():
        return {
            "document_type": "unknown",
            "official_name": "",
            "confidence":    "low",
            "notes": (
                "Tesseract returned no text — image may be too blurry, "
                "dark, or rotated. Try a clearer photo in good lighting."
            ),
            "raw_text": "",
        }

    doc_type, confidence = _detect_doc_type(full_text)
    official_name        = _extract_name(full_text, doc_type)

    if not official_name and confidence == "high":
        confidence = "medium"

    notes = "; ".join(ocr_errors) if ocr_errors else "None"

    return {
        "document_type": doc_type,
        "official_name": official_name,
        "confidence":    confidence,
        "notes":         notes,
        "raw_text":      full_text,   # exposed in debug expander
    }


# ================================================================
#              DETERMINISTIC BASELINE
# ================================================================

def expected_decision_local(user: dict) -> str:
    has_aadhaar    = "aadhaar"  in user["documents"]
    has_pan        = "pan"      in user["documents"]
    has_substitute = any(d in user["documents"] for d in ["passport", "voter_id"])

    if not (has_aadhaar and has_pan) and not has_substitute:
        return "REJECTED"
    if not has_aadhaar and not has_substitute:
        return "REJECTED"
    if user.get("name", "").lower().strip() != user.get("submitted_name", "").lower().strip():
        return "REJECTED"
    if _parse_income(user.get("income", 0)) < 100000:
        return "MANUAL REVIEW"
    return "APPROVED"


# ================================================================
#                          UI LAYOUT
# ================================================================

left, right = st.columns([1, 1.5])

# ── LEFT panel ────────────────────────────────────────────────── #
with left:
    st.header("📋 Applicant Data")

    st.subheader("Step 1 — Upload Identity Documents")
    st.caption(
        "Upload Aadhaar, PAN, Passport, or Voter ID (JPG / PNG / PDF).  \n"
        "The **official name** is extracted automatically via local OCR — no internet required."
    )

    uploaded_files = st.file_uploader(
        "Drop document images here",
        type=["jpg", "jpeg", "png", "webp", "pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    extracted_docs  = []
    extracted_names = []
    ocr_rows        = []

    if uploaded_files:
        with st.spinner("🔍 Running Tesseract OCR locally…"):
            for uf in uploaded_files:
                info        = extract_info_from_document(uf)
                doc_type    = info["document_type"]
                name_on_doc = info["official_name"]
                confidence  = info["confidence"]
                notes       = info["notes"]
                raw_text    = info.get("raw_text", "")

                if doc_type != "unknown":
                    extracted_docs.append(doc_type)
                if name_on_doc:
                    extracted_names.append(name_on_doc)

                label     = DOC_LABEL_MAP.get(doc_type, "Unknown Document")
                conf_icon = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(confidence, "⚪")

                ocr_rows.append({
                    "file":       uf.name,
                    "label":      label,
                    "name":       name_on_doc or "—",
                    "confidence": f"{conf_icon} {confidence}",
                    "notes":      notes,
                    "raw_text":   raw_text,
                })

        # ── Extraction result cards ──────────────────────────── #
        st.markdown("**Extraction Results:**")
        for row in ocr_rows:
            with st.container(border=True):
                col_a, col_b = st.columns([1, 1])
                with col_a:
                    st.markdown(f"📄 `{row['file']}`")
                    st.markdown(f"**Type detected:** {row['label']}")
                    st.markdown(f"**Confidence:** {row['confidence']}")
                with col_b:
                    st.markdown("**Name on document:**")
                    st.code(row["name"])
                    if row["notes"] and row["notes"].lower() != "none":
                        st.caption(f"⚠️ {row['notes']}")

                # Raw OCR text — useful for debugging / research
                with st.expander("🔬 View raw OCR text (research / debug)"):
                    st.text(row["raw_text"] or "(empty)")

        if extracted_docs:
            st.success(
                "✅ Detected: "
                + ", ".join(DOC_LABEL_MAP.get(d, d) for d in extracted_docs)
            )
        else:
            st.warning(
                "⚠️ Document type could not be detected automatically.  \n"
                "Please select it manually below, and type the official name."
            )

    st.divider()

    # ── Step 2: confirm / edit ───────────────────────────────── #
    st.subheader("Step 2 — Confirm Details")

    default_name = extracted_names[0] if extracted_names else ""
    name = st.text_input(
        "Official Name *(auto-extracted — edit if wrong)*",
        value=default_name,
        placeholder="Will be filled after upload"
    )

    submitted = st.text_input(
        "Submitted Name *(as written by the applicant on the KYC form)*",
        value="",
        placeholder="e.g.  Ravi Kumar"
    )

    income = st.number_input(
        "Annual Income (₹)",
        min_value=0,
        value=500000,
        step=10000
    )

    st.caption("Detected document types *(correct if needed)*:")
    docs = st.multiselect(
        "Documents present",
        options=["aadhaar", "pan", "passport", "voter_id"],
        default=list(dict.fromkeys(extracted_docs)),
        label_visibility="collapsed"
    )

    st.divider()
    verify = st.button(
        "🔍 Run KYC Verification",
        use_container_width=True,
        type="primary",
        disabled=not bool(uploaded_files)
    )

# ── RIGHT panel ───────────────────────────────────────────────── #
with right:
    st.header("🤖 Agent Activity Log")
    activity_box = st.empty()

    if not uploaded_files:
        activity_box.info(
            "👈  Upload at least one identity document on the left, "
            "then click **Run KYC Verification**."
        )


# ================================================================
#                      RUN VERIFICATION
# ================================================================

if verify:
    if not docs:
        st.warning("⚠️ No document types selected. Add them in the multiselect above.")
        st.stop()
    if not name.strip():
        st.warning(
            "⚠️ Official name is empty. "
            "OCR could not read it — please type it manually in the field above."
        )
        st.stop()
    if not submitted.strip():
        st.warning("⚠️ Please enter the submitted name (as written by the applicant).")
        st.stop()

    user = {
        "id":             "UI",
        "name":           name.strip(),
        "submitted_name": submitted.strip(),
        "documents":      docs,
        "income":         income,
    }

    activity = "Starting KYC verification pipeline…\n\n"
    activity_box.code(activity)

    try:
        document_task, identity_task, risk_task, compliance_task = create_tasks(user)

        crew = Crew(
            agents=[document_agent, identity_agent, risk_agent, compliance_agent],
            tasks=[document_task, identity_task, risk_task, compliance_task],
            verbose=False,
            memory=False
        )

        activity += "🤖 Document Agent    → running…\n"
        activity_box.code(activity)

        result       = crew.kickoff()
        output       = str(result)
        output_upper = output.upper()

        activity += "✅ Document Agent    → done\n"
        activity += "✅ Identity Agent    → done\n"
        activity += "✅ Risk Agent        → done\n"
        activity += "✅ Compliance Agent  → done\n\n"

        # Parse decision
        if "APPROVED"   in output_upper:   decision = "APPROVED"
        elif "REJECT"   in output_upper:   decision = "REJECTED"
        elif "MANUAL"   in output_upper:   decision = "MANUAL REVIEW"
        else:                              decision = "UNKNOWN"

        baseline    = expected_decision_local(user)
        match_label = "✅ Matches baseline" if decision == baseline else "⚠️ Deviates from baseline"
        name_match  = name.strip().lower() == submitted.strip().lower()
        income_int  = _parse_income(income)

        summary = (
            "=================================\n"
            "        FINAL KYC RESULT\n"
            "=================================\n"
            f"Official Name  : {name}\n"
            f"Submitted Name : {submitted}\n"
            f"Name Match     : {'Yes' if name_match else 'No'}\n"
            f"Documents      : {', '.join(docs)}\n"
            f"Income         : Rs.{income_int:,}\n"
            f"\nDecision       : {decision}\n"
            f"Baseline       : {baseline}  ({match_label})\n"
            "=================================\n\n"
            "--- Agent Reasoning ---\n"
            f"{output}"
        )

        activity += summary
        activity_box.code(activity)

        if decision == "APPROVED":
            st.success("✅ KYC APPROVED")
        elif decision == "REJECTED":
            st.error("❌ KYC REJECTED")
        elif decision == "MANUAL REVIEW":
            st.warning("⚠️ MANUAL REVIEW REQUIRED")
        else:
            st.info("❓ DECISION UNKNOWN — check agent output above")

        if decision != baseline:
            st.info(
                f"🔬 **Research Note:** Agent decision **{decision}** deviates from "
                f"deterministic baseline **{baseline}**. "
                "This may indicate emergent reasoning — review the agent log."
            )

    except Exception as e:
        st.error(f"❌ System Error: {e}")
        with st.expander("🔍 Full Traceback (for debugging)"):
            st.code(traceback.format_exc())