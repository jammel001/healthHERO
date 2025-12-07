import os
import io
import csv
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

from flask import (
    Flask, request, jsonify, render_template, session, send_file
)
from flask_cors import CORS
from werkzeug.middleware.proxy_fix import ProxyFix
from flask_session import Session
from werkzeug.security import generate_password_hash, check_password_hash

# DB
from sqlalchemy import (
    create_engine, Column, String, Integer, DateTime, Text, Float, Boolean, ForeignKey
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

# ML + utils
import joblib
import numpy as np
from rapidfuzz import process, fuzz
import requests
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import mm

# ---------------------------
# Config
# ---------------------------
app = Flask(__name__)
CORS(app)
app.wsgi_app = ProxyFix(app.wsgi_app)

app.secret_key = os.environ.get("FLASK_SECRET_KEY", "change-this-secret")

app.config['SESSION_TYPE'] = "filesystem"
app.config['SESSION_FILE_DIR'] = './flask_session'
app.config['SESSION_PERMANENT'] = False
Session(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------
# FILE PATHS
# ---------------------------
MODEL_TABLES = os.path.join(BASE_DIR, "model_tables.pkl")
SYM_EMB = os.path.join(BASE_DIR, "symptom_embeddings.npz")
SYM_ENCODER = os.path.join(BASE_DIR, "symptom_encoder.pkl")
SYM_EXPLANATION = os.path.join(BASE_DIR, "symptom_to_explanation.pkl")
DISEASE_MODEL = os.path.join(BASE_DIR, "disease_model.pkl")
DISEASE_DESC = os.path.join(BASE_DIR, "disease_to_description.pkl")
DISEASE_PRECAUTIONS = os.path.join(BASE_DIR, "disease_to_precautions.pkl")
LABEL_ENCODER = os.path.join(BASE_DIR, "label_encoder.pkl")

DB_URL = os.environ.get("DATABASE_URL", "sqlite:///careguide.db")
ADMIN_KEY = os.environ.get("ADMIN_KEY", "change-this-admin-key")

# ---------------------------
# Database
# ---------------------------
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
# LOAD ALL MODEL FILES
# ---------------------------
def safe_load(path):
    try:
        return joblib.load(path)
    except Exception as e:
        print(f"ERROR loading {path}:", e)
        return None


def safe_numpy(path):
    try:
        return np.load(path, allow_pickle=True)
    except Exception as e:
        print(f"ERROR loading {path}:", e)
        return None


model_tables = safe_load(MODEL_TABLES)
symptom_encoder = safe_load(SYM_ENCODER)
symptom_explanations = safe_load(SYM_EXPLANATION)
disease_model = safe_load(DISEASE_MODEL)
disease_descriptions = safe_load(DISEASE_DESC)
disease_precautions = safe_load(DISEASE_PRECAUTIONS)
label_encoder = safe_load(LABEL_ENCODER)
symptom_embeddings = safe_numpy(SYM_EMB)

# ---------------------------
# MODEL BUNDLE
# ---------------------------
class ModelBundle:
    class ModelBundle:
    def __init__(self):
        self.model = disease_model
        self.label_encoder = label_encoder
        self.symptoms = list(symptom_encoder.classes_) if symptom_encoder is not None else []

        if symptom_embeddings is not None:
            self.embedding_vocab = list(symptom_embeddings.files)
            self.embedding_matrix = symptom_embeddings[self.embedding_vocab[0]]
        else:
            self.embedding_vocab = []
            self.embedding_matrix = None

        return list(set(matched))

    def predict(self, matched: List[str]) -> Dict[str, Any]:
        if not matched or self.model is None:
            return {"condition": "Unknown", "prob": 0, "advice": "Model unavailable"}

        X = np.zeros((1, len(self.symptoms)))
        for sym in matched:
            if sym in self.symptoms:
                idx = self.symptoms.index(sym)
                X[0, idx] = 1

        probs = self.model.predict_proba(X)[0]
        idx = np.argmax(probs)
        label = self.label_encoder.inverse_transform([idx])[0]
        prob = float(probs[idx])

        desc = disease_descriptions.get(label, "No description")
        precautions = disease_precautions.get(label, [])

        return {
            "condition": label,
            "prob": prob,
            "description": desc,
            "precautions": precautions
        }

BUNDLE = ModelBundle()

# ---------------------------
# Routes (HTML pages)
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
    if not data:
        return render_template("result.html", result=None)
    return render_template("result.html", result=data)


# ---------------------------
# API
# ---------------------------
def parse_tokens(text: str) -> List[str]:
    return [t.strip().lower() for t in text.replace(';', ',').split(',') if t.strip()]


@app.route("/api/diagnose", methods=["POST"])
def diagnose():
    data = request.json or {}
    symptoms_raw = data.get("symptoms", "")
    name = data.get("name", "Anonymous")

    tokens = parse_tokens(symptoms_raw)
    matched = BUNDLE.match_symptoms(tokens)
    result = BUNDLE.predict(matched)

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

    session["last_result"] = result

    return jsonify({
        "condition": result["condition"],
        "probability": result["prob"],
        "description": result["description"],
        "precautions": result["precautions"]
    })


# ---------------------------
# PDF GENERATION
# ---------------------------
@app.route('/download_pdf')
def download_pdf():
    last = session.get('last_result')
    if not last:
        return "No result", 400

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)

    y = A4[1] - 20 * mm
    c.drawString(20 * mm, y, "HealthChero Medical Report")
    y -= 15

    c.drawString(20 * mm, y, f"Condition: {last['condition']}")
    y -= 10
    c.drawString(20 * mm, y, f"Probability: {last['prob']:.2f}")
    y -= 10

    c.drawString(20 * mm, y, "Description:")
    y -= 10
    for line in last['description'].split(". "):
        c.drawString(20 * mm, y, line.strip())
        y -= 7

    y -= 10
    c.drawString(20 * mm, y, "Precautions:")
    y -= 10
    for p in last['precautions']:
        c.drawString(25 * mm, y, f"- {p}")
        y -= 7

    c.save()
    buf.seek(0)

    return send_file(buf, as_attachment=True,
                     download_name="health_report.pdf",
                     mimetype="application/pdf")


# ---------------------------
# Health
# ---------------------------
@app.route("/health")
def health():
    return jsonify({
        "model": bool(disease_model),
        "encoder": bool(label_encoder),
        "embeddings": bool(symptom_embeddings)
    })


# ---------------------------
# RUN
# ---------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0",
            port=int(os.environ.get("PORT", 10000)),
            debug=False)
