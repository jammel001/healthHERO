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

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import mm

# ---------------------------
# Config
# ---------------------------
app = Flask(__name__, template_folder='templates')
CORS(app)
app.wsgi_app = ProxyFix(app.wsgi_app)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "change-this-secret")
app.config['SESSION_TYPE'] = "filesystem"
app.config['SESSION_FILE_DIR'] = './flask_session'
app.config['SESSION_PERMANENT'] = False
Session(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------
# File Paths
# ---------------------------
FILES = {
    "model_tables": "model_tables.pkl",
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
        print(f"ERROR loading {path}: {e}")
        return None

def safe_numpy(path):
    try:
        return np.load(path, allow_pickle=True)
    except Exception as e:
        print(f"ERROR loading {path}: {e}")
        return None

# Load files
model_tables = safe_load(os.path.join(BASE_DIR, FILES["model_tables"]))
symptom_embeddings = safe_numpy(os.path.join(BASE_DIR, FILES["symptom_embeddings"]))
symptom_encoder = safe_load(os.path.join(BASE_DIR, FILES["symptom_encoder"]))
symptom_explanations = safe_load(os.path.join(BASE_DIR, FILES["symptom_explanations"]))
disease_model = safe_load(os.path.join(BASE_DIR, FILES["disease_model"]))
disease_descriptions = safe_load(os.path.join(BASE_DIR, FILES["disease_descriptions"]))
disease_precautions = safe_load(os.path.join(BASE_DIR, FILES["disease_precautions"]))
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
    user_id = Column(Integer, ForeignKey('users.id'), nullable=True)
    symptoms_entered = Column(Text)
    matched_symptoms = Column(Text)
    predicted_condition = Column(String(256))
    probability = Column(Float)
    advice = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    user = relationship("User", backref="assessments")

Base.metadata.create_all(engine)

# ---------------------------
# Model Bundle
# ---------------------------
class ModelBundle:
    def __init__(self):
        self.model = disease_model
        self.label_encoder = label_encoder
        self.symptom_encoder = symptom_encoder
        self.symptoms = symptom_encoder.get_feature_names_out().tolist() if symptom_encoder else []
        self.embedding_keys = list(symptom_embeddings.files) if symptom_embeddings else []
        self.embedding_matrix = symptom_embeddings[self.embedding_keys[0]] if symptom_embeddings else None

    def match_symptoms(self, symptoms: List[str]) -> List[str]:
        matched = []
        for s in symptoms:
            if s in self.symptoms:
                matched.append(s)
            else:
                suggestions = process.extract(s, self.symptoms, limit=1, score_cutoff=80)
                if suggestions:
                    matched.append(suggestions[0][0])
        return list(set(matched))

    def predict(self, matched: List[str]) -> dict:
        if not matched or self.model is None:
            return {
                "condition": "Unknown",
                "prob": 0.0,
                "description": "Model unavailable",
                "precautions": []
            }
        symptom_text = " ".join(matched)
        X = self.symptom_encoder.transform([symptom_text])
        probs = self.model.predict_proba(X)[0]
        idx = int(np.argmax(probs))
        label = self.label_encoder.inverse_transform([idx])[0]
        prob = float(probs[idx])
        desc = disease_descriptions.get(label, "No description available")
        precautions = disease_precautions.get(label, [])
        return {"condition": label, "prob": prob, "description": desc, "precautions": precautions}

BUNDLE = ModelBundle()

# ---------------------------
# Utilities
# ---------------------------
def parse_tokens(text: str) -> List[str]:
    return [t.strip().lower() for t in text.replace(';', ',').split(',') if t.strip()]

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
    data = session.get("last_result")
    return render_template("result.html", result=data)

# ---------------------------
# Diagnosis API
# ---------------------------
@app.route("/api/diagnose", methods=["POST"])
def diagnose():
    data = request.json or {}
    name = data.get("name", "Anonymous")
    tokens = parse_tokens(data.get("symptoms", ""))

    matched = BUNDLE.match_symptoms(tokens)
    result = BUNDLE.predict(matched)

    # Per-symptom explanations
    explanations = []
    for s in matched:
        expl = symptom_explanations.get(s, "No explanation available")
        explanations.append(f"{s}: {expl}")

    # Save to DB
    db = DBSession()
    ass = Assessment(
        user_id=session.get("user_id"),
        symptoms_entered="; ".join(tokens),
        matched_symptoms="; ".join(matched),
        predicted_condition=result["condition"],
        probability=result["prob"],
        advice=result["description"]
    )
    db.add(ass)
    db.commit()

    session["last_result"] = {
        "condition": result["condition"],
        "probability": result["prob"],
        "description": result["description"],
        "precautions": result["precautions"],
        "explanations": explanations
    }

    return jsonify(session["last_result"])

# ---------------------------
# PDF Generation
# ---------------------------
@app.route('/download_pdf')
def download_pdf():
    last = session.get('last_result')
    if not last:
        return "No result available", 400

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    y = A4[1] - 20*mm

    c.setFont("Helvetica-Bold", 14)
    c.drawString(20*mm, y, "HealthChero Medical Report")
    y -= 15

    c.setFont("Helvetica", 12)
    c.drawString(20*mm, y, f"Condition: {last['condition']}")
    y -= 10
    c.drawString(20*mm, y, f"Probability: {last['probability']:.2f}")
    y -= 10
    c.drawString(20*mm, y, "Description:")
    y -= 10
    for line in last['description'].split(". "):
        c.drawString(20*mm, y, line.strip())
        y -= 7
    y -= 10

    if last.get('precautions'):
        c.drawString(20*mm, y, "Precautions:")
        y -= 10
        for p in last['precautions']:
            c.drawString(25*mm, y, f"- {p}")
            y -= 7
        y -= 10

    if last.get('explanations'):
        c.drawString(20*mm, y, "Symptom Explanations:")
        y -= 10
        for e in last['explanations']:
            c.drawString(25*mm, y, f"- {e}")
            y -= 7

    c.save()
    buf.seek(0)
    return send_file(buf, as_attachment=True, download_name="health_report.pdf", mimetype="application/pdf")

# ---------------------------
# Health Check
# ---------------------------
@app.route("/health")
def health():
    return jsonify({
        "model": bool(disease_model),
        "encoder": bool(label_encoder),
        "embeddings": bool(symptom_embeddings)
    })

# ---------------------------
# Run
# ---------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)), debug=False)
