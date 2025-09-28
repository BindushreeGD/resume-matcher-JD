import os
import re
import io
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer, util
from typing import Tuple
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
from pdfminer.high_level import extract_text as pdfminer_extract_text  # pdfminer.six

app = Flask(__name__, template_folder="templates")
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"pdf"}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB

# SBERT Model
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
model = SentenceTransformer(MODEL_NAME)

# Keywords & Patterns
POS_SECTION_HEADERS = {
    "education", "work experience", "experience", "professional experience",
    "projects", "skills", "technical skills", "summary", "profile",
    "certifications", "achievements", "publications", "responsibilities",
    "languages", "interests"
}
DEGREE_KEYWORDS = {"b.e", "btech", "b.tech", "mtech", "m.tech", "bsc", "b.sc",
                   "msc", "m.sc", "mba", "bca", "mca", "phd"}
NEG_KEYWORDS = {
    "offer", "invoice", "policy", "quotation", "purchase order", "receipt",
    "terms and conditions", "agreement", "nda", "contract", "gst", "bill to",
    "ship to", "tax", "pan", "aadhaar", "bank account"
}

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"(\+?\d[\d\-\s()]{7,}\d)")
LINKEDIN_RE = re.compile(r"(linkedin\.com/in/)")
GITHUB_RE = re.compile(r"(github\.com/)")
PORTFOLIO_RE = re.compile(r"(https?://[^\s]+)")
YEAR_RANGE_RE = re.compile(r"(20\d{2}\s*[-–]\s*20\d{2})")
YEAR_SINGLE_RE = re.compile(r"(20\d{2})")
BULLET_RE = re.compile(r"(^|\n)\s*[\-\•\●\▪\▶\*]\s+", re.MULTILINE)


# -------------------------------
# Helpers
# -------------------------------
def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def extract_text_from_pdf(pdf_path: str) -> str:
    """Try extracting text using PyMuPDF → pdfminer.six → PyPDF2 → OCR fallback."""
    text = ""

    # 1. Try PyMuPDF
    try:
        doc = fitz.open(pdf_path)
        text = "\n".join(page.get_text("text") for page in doc)
        doc.close()
        if text.strip():
            return text
    except Exception as e:
        print(f"[WARN] PyMuPDF failed: {e}")

    # 2. Try pdfminer.six
    try:
        text = pdfminer_extract_text(pdf_path)
        if text and text.strip():
            return text
    except Exception as e:
        print(f"[WARN] pdfminer.six failed: {e}")

    # 3. Try PyPDF2
    try:
        reader = PdfReader(pdf_path)
        text_parts = [page.extract_text() or "" for page in reader.pages]
        text = "\n".join(text_parts).strip()
        if text.strip():
            return text
    except Exception as e:
        print(f"[WARN] PyPDF2 failed: {e}")

    # 4. OCR with Tesseract
    try:
        doc = fitz.open(pdf_path)
        ocr_texts = []
        for page_num in range(len(doc)):
            pix = doc[page_num].get_pixmap()
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            ocr_text = pytesseract.image_to_string(img)
            ocr_texts.append(ocr_text)
        doc.close()
        text = "\n".join(ocr_texts).strip()
        if text.strip():
            return text
    except Exception as e:
        print(f"[ERROR] OCR failed: {e}")

    return text.strip()


def extract_metadata_text(pdf_path: str) -> str:
    try:
        reader = PdfReader(pdf_path)
        meta = reader.metadata
        if not meta:
            return ""
        return " ".join(str(v) for v in meta.values() if v)
    except Exception:
        return ""


def _count_bullets(text: str) -> int:
    return len(BULLET_RE.findall(text))


def _is_contact_list_like(text: str, section_hits: int, degree_hits: int) -> bool:
    """Avoid misclassifying contact directories as resumes."""
    emails = EMAIL_RE.findall(text)
    phones = PHONE_RE.findall(text)
    return len(emails) >= 5 and len(phones) >= 5 and section_hits == 0 and degree_hits == 0


def looks_like_resume(text: str) -> Tuple[bool, str, bool]:
    """Check if the text looks like a resume."""
    if not text:
        return (False, "Empty or unreadable PDF.", False)

    t = text.lower()
    words = len(t.split())
    if words < 50:  # relaxed threshold
        return (False, "Too little text to be a resume.", False)

    has_email = EMAIL_RE.search(t) is not None
    has_phone = PHONE_RE.search(t) is not None
    section_hits = sum(1 for s in POS_SECTION_HEADERS if s in t)
    degree_hits = sum(1 for d in DEGREE_KEYWORDS if d in t)

    if _is_contact_list_like(t, section_hits, degree_hits):
        return (False, "Looks like a contacts/directory list, not a resume.", False)

    has_bullets = _count_bullets(text) >= 1
    has_year_range = YEAR_RANGE_RE.search(t) is not None
    has_enough_years = len(YEAR_SINGLE_RE.findall(t)) >= 1

    neg_hits = sum(1 for k in NEG_KEYWORDS if k in t)
    has_negatives = neg_hits > 0

    if (has_email or has_phone) and (section_hits >= 1 or degree_hits >= 1 or has_year_range or has_enough_years):
        return (True, "Looks like a resume.", has_negatives)

    return (False, "Does not match resume structure.", has_negatives)


def sbert_similarity_percent(jd: str, resume_text: str) -> float:
    job_emb = model.encode(jd, normalize_embeddings=True)
    res_emb = model.encode(resume_text, normalize_embeddings=True)
    sim = float(util.cos_sim(job_emb, res_emb)[0][0])
    return round(max(0.0, min(1.0, sim)) * 100.0, 2)


# -------------------------------
# Routes
# -------------------------------
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    if "resumes" not in request.files or "job_description" not in request.form:
        return render_template("result.html", error="Please provide a Job Description and at least one PDF resume.")

    jd = (request.form.get("job_description") or "").strip()
    if not jd:
        return render_template("result.html", error="Job Description cannot be empty.")

    files = request.files.getlist("resumes")
    if not files:
        return render_template("result.html", error="No files selected.")

    results = []
    for file in files:
        if file.filename == "":
            continue
        if not allowed_file(file.filename):
            results.append({"filename": file.filename, "percentage": None,
                            "label": "Invalid file type", "note": "Only PDF files are allowed."})
            continue

        save_path = os.path.join(app.config["UPLOAD_FOLDER"], secure_filename(file.filename))
        file.save(save_path)

        text = extract_text_from_pdf(save_path)
        metadata_text = extract_metadata_text(save_path)

        found_hidden = [k for k in NEG_KEYWORDS if re.search(rf"\b{k}\b", metadata_text, re.IGNORECASE)]
        is_resume, note, has_negatives = looks_like_resume(text)

        if not is_resume:
            if found_hidden:
                note += " (Hidden metadata contains non-resume terms.)"
            results.append({"filename": file.filename, "percentage": None, "label": "Not a resume", "note": note})
            continue

        percent = sbert_similarity_percent(jd, text)
        if has_negatives:
            percent = max(0, percent - 10)

        label = "Strong match" if percent >= 70 else "Moderate match" if percent >= 50 else "Weak match"
        if found_hidden:
            note += f" (Hidden metadata: {', '.join(found_hidden)})"

        results.append({"filename": file.filename, "percentage": percent, "label": label, "note": note})

    return render_template("result.html", results=results)


if __name__ == "__main__":
    app.run(debug=True)
