import os
import io
from datetime import datetime
from typing import List

from flask import Flask, request, jsonify, render_template, session, send_file
from flask_cors import CORS
from werkzeug.middleware.proxy_fix import ProxyFix
from flask_session import Session

from sqlalchemy import create_engine, Column, String, Integer, DateTime, Text, Float, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

import joblib
import numpy as np
from rapidfuzz import process

from sklearn.metrics.pairwise import cosine_similarity

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import mm

# ---------------------------
# App Config
# ---------------------------
app = Flask(__name__, template_folder="templates")
CORS(app)
app.wsgi_app = ProxyFix(app.wsgi_app)

app.secret_key = os.environ.get("FLASK_SECRET_KEY", "change-this-secret")
app.config["SESSION_TYPE"] = "filesystem"
app.config["SESSION_FILE_DIR"] = "./flask_session"
app.config["SESSION_PERMANENT"] = False
Session(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------
# File Paths
# ---------------------------
FILES = {
    "symptom_embeddings": "symptom_embeddings.npz",
    "symptom_encoder": "symptom_encoder.pkl",
    "symptom_explanations": "symptom_to_explanation.pkl",
    "disease_model": "disease_model.pkl",
    "disease_descriptions": "disease_to_description.pkl",
    "disease_precautions": "disease_to_precautions.pkl",
    "label_encoder": "label_encoder.pkl"
}

def safe_load(path):
    try:
        return joblib.load(path)
    except Exception as e:
        print(f"[WARN] Could not load {path}: {e}")
        return None

def safe_numpy(path):
    try:
        return np.load(path, allow_pickle=True)
    except Exception as e:
        print(f"[WARN] Could not load {path}: {e}")
        return None

# ---------------------------
# Load Models
# ---------------------------
symptom_encoder = safe_load(os.path.join(BASE_DIR, FILES["symptom_encoder"]))
symptom_embeddings = safe_numpy(os.path.join(BASE_DIR, FILES["symptom_embeddings"]))
symptom_explanations = safe_load(os.path.join(BASE_DIR, FILES["symptom_explanations"])) or {}
disease_model = safe_load(os.path.join(BASE_DIR, FILES["disease_model"]))
disease_descriptions = safe_load(os.path.join(BASE_DIR, FILES["disease_descriptions"])) or {}
disease_precautions = safe_load(os.path.join(BASE_DIR, FILES["disease_precautions"])) or {}
label_encoder = safe_load(os.path.join(BASE_DIR, FILES["label_encoder"]))

# ---------------------------
# Database
# ---------------------------
DB_URL = os.environ.get("DATABASE_URL", "sqlite:///careguide.db")
engine = create_engine(DB_URL, echo=False, future=True)
Base = declarative_base()
DBSession = sessionmaker(bind=engine)

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    email = Column(String(256), unique=True, nullable=False)
    password_hash = Column(String(512), nullable=False)
    display_name = Column(String(128))
    created_at = Column(DateTime, default=datetime.utcnow)

class Assessment(Base):
    __tablename__ = "assessments"
    id = Column(Integer, primary_key=True)
    symptoms_entered = Column(Text)
    matched_symptoms = Column(Text)
    predicted_condition = Column(String(256))
    probability = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(engine)

# ---------------------------
# Model Bundle
# ---------------------------
class ModelBundle:
    def __init__(self):
        self.model = disease_model
        self.encoder = symptom_encoder
        self.label_encoder = label_encoder

        self.symptoms = (
            self.encoder.get_feature_names_out().tolist()
            if self.encoder else []
        )

        if symptom_embeddings is not None:
            self.embedding_keys = list(symptom_embeddings.files)
            self.embedding_matrix = symptom_embeddings[self.embedding_keys[0]]
        else:
            self.embedding_matrix = None

    def match_symptoms(self, symptoms: List[str]) -> List[str]:
        matched = []

        for s in symptoms:
            if s in self.symptoms:
                matched.append(s)
                continue

            fuzzy = process.extract(s, self.symptoms, limit=1, score_cutoff=80)
            if fuzzy:
                matched.append(fuzzy[0][0])
                continue

            if self.embedding_matrix is not None:
                try:
                    vec = self.encoder.transform([s]).toarray()
                    sims = cosine_similarity(vec, self.embedding_matrix)[0]
                    best = int(np.argmax(sims))
                    matched.append(self.symptoms[best])
                except Exception:
                    pass

        return list(set(matched))

    def predict_topk(self, matched: List[str], k: int = 3):
        if not matched or self.model is None:
            return []

        text = " ".join(matched)
        X = self.encoder.transform([text])
        probs = self.model.predict_proba(X)[0]

        idxs = np.argsort(probs)[::-1][:k]
        results = []

        for i in idxs:
            label = self.label_encoder.inverse_transform([i])[0]
            results.append({
                "condition": label,
                "probability": float(probs[i]),
                "description": disease_descriptions.get(label, ""),
                "precautions": disease_precautions.get(label, [])
            })

        return results

# ✅ INITIALIZE BUNDLE (CRITICAL FIX)
BUNDLE = ModelBundle()

# ---------------------------
# Utilities
# ---------------------------
def parse_tokens(text: str) -> List[str]:
    return [t.strip().lower() for t in text.replace(";", ",").split(",") if t.strip()]

# ---------------------------
# Routes
# ---------------------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/guidelines")
def guidelines():
    return render_template("guidelines.html")

@app.route("/result")
def result_page():
    return render_template("result.html", result=session.get("last_result"))

# ---------------------------
# Diagnosis API
# ---------------------------
@app.route("/api/diagnose", methods=["POST"])
def diagnose():
    data = request.json or {}
    tokens = parse_tokens(data.get("symptoms", ""))

    matched = BUNDLE.match_symptoms(tokens)
    predictions = BUNDLE.predict_topk(matched)

    explanations = [
        f"{s}: {symptom_explanations.get(s, 'No explanation available.')}"
        for s in matched
    ]

    if not predictions:
        return jsonify({"error": "Prediction failed", "explanations": explanations}), 400

    top = predictions[0]
    warning = None
    if top["probability"] < 0.35:
        warning = "⚠️ Low confidence result. Consult a healthcare professional."

    result = {
        "top_prediction": top,
        "other_predictions": predictions[1:],
        "confidence_warning": warning,
        "explanations": explanations
    }

    session["last_result"] = result
    return jsonify(result)

# ---------------------------
# PDF Download
# ---------------------------
@app.route("/download_pdf")
def download_pdf():
    last = session.get("last_result")
    if not last:
        return "No result available", 400

    top = last["top_prediction"]

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    y = A4[1] - 20 * mm

    c.setFont("Helvetica-Bold", 14)
    c.drawString(20 * mm, y, "HealthChero Medical Report")
    y -= 20

    c.setFont("Helvetica", 12)
    c.drawString(20 * mm, y, f"Condition: {top['condition']}")
    y -= 12
    c.drawString(20 * mm, y, f"Probability: {top['probability']:.2f}")
    y -= 12

    c.drawString(20 * mm, y, "Description:")
    y -= 10
    for line in top["description"].split(". "):
        c.drawString(25 * mm, y, line.strip())
        y -= 8

    if top["precautions"]:
        y -= 10
        c.drawString(20 * mm, y, "Precautions:")
        y -= 10
        for p in top["precautions"]:
            c.drawString(25 * mm, y, f"- {p}")
            y -= 8

    if last["explanations"]:
        y -= 10
        c.drawString(20 * mm, y, "Symptom Explanations:")
        y -= 10
        for e in last["explanations"]:
            c.drawString(25 * mm, y, f"- {e}")
            y -= 8

    c.save()
    buf.seek(0)

    return send_file(
        buf,
        as_attachment=True,
        download_name="health_report.pdf",
        mimetype="application/pdf"
    )

# ---------------------------
# Health Check
# ---------------------------
@app.route("/health")
def health():
    return jsonify({"status": "ok"}), 200

# ---------------------------
# Run
# ---------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)), debug=False)
