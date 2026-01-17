import os
import io
from datetime import datetime
from typing import List

from flask import Flask, request, jsonify, render_template, session, send_file
from flask_cors import CORS
from werkzeug.middleware.proxy_fix import ProxyFix
from flask_session import Session

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
# Load Models
# ---------------------------
def safe_load(path):
    try:
        return joblib.load(path)
    except Exception:
        return None

def safe_numpy(path):
    try:
        return np.load(path, allow_pickle=True)
    except Exception:
        return None

symptom_encoder = safe_load(os.path.join(BASE_DIR, "symptom_encoder.pkl"))
symptom_embeddings = safe_numpy(os.path.join(BASE_DIR, "symptom_embeddings.npz"))
symptom_explanations = safe_load(os.path.join(BASE_DIR, "symptom_to_explanation.pkl")) or {}
disease_model = safe_load(os.path.join(BASE_DIR, "disease_model.pkl"))
disease_descriptions = safe_load(os.path.join(BASE_DIR, "disease_to_description.pkl")) or {}
disease_precautions = safe_load(os.path.join(BASE_DIR, "disease_to_precautions.pkl")) or {}
label_encoder = safe_load(os.path.join(BASE_DIR, "label_encoder.pkl"))

# ---------------------------
# Symptom intelligence
# ---------------------------
SYMPTOM_ALIASES = {
    "hot body": "fever",
    "stomach headache": "abdominal pain",
    "throwing up": "vomiting",
    "weak body": "fatigue",
    "loss of food taste": "loss of appetite"
}

KNOWN_SYMPTOMS = set(symptom_explanations.keys())

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
            self.embedding_matrix = symptom_embeddings[list(symptom_embeddings.files)[0]]
        else:
            self.embedding_matrix = None

    def match_symptoms(self, symptoms):
        matched, clarifications = [], []

        for s in symptoms:
            if s in SYMPTOM_ALIASES:
                clarifications.append(
                    f"When you said '{s}', did you mean '{SYMPTOM_ALIASES[s]}'?"
                )
                matched.append(SYMPTOM_ALIASES[s])
                continue

            if s in self.symptoms:
                matched.append(s)
                continue

            fuzzy = process.extract(s, self.symptoms, limit=1, score_cutoff=80)
            if fuzzy:
                clarifications.append(
                    f"Did you mean '{fuzzy[0][0]}' instead of '{s}'?"
                )
                matched.append(fuzzy[0][0])

        return list(set(matched)), clarifications

    def predict(self, matched):
        if not matched or not self.model:
            return []

        X = self.encoder.transform([" ".join(matched)])
        probs = self.model.predict_proba(X)[0]
        idxs = np.argsort(probs)[::-1][:3]

        results = []
        for i in idxs:
            label = self.label_encoder.inverse_transform([i])[0]
            results.append({
                "condition": label,
                "probability": round(float(probs[i]), 2),
                "description": disease_descriptions.get(label, ""),
                "precautions": disease_precautions.get(label, [])
            })
        return results

BUNDLE = ModelBundle()

# ---------------------------
# Routes
# ---------------------------
@app.route("/")
def home():
    return render_template("index.html")

# ---------------------------
# Diagnosis API
# ---------------------------
@app.route("/api/diagnose", methods=["POST"])
def diagnose():
    data = request.json or {}
    msg = (data.get("message") or "").strip().lower()

    # Restart
    if msg == "__restart__":
        session.clear()

    stage = session.get("stage", "GREETING")

  # -------------------------------
# ASK SYMPTOMS
# -------------------------------
if stage == "ASK_SYMPTOMS":
    tokens = parse_tokens(user_input)

    if not tokens:
        return jsonify({
            "type": "message",
            "text": "Please tell me at least one symptom (for example: fever, headache, vomiting)."
        })

    matched = BUNDLE.match_symptoms(tokens)

    # 🚑 Handle unrecognized / wrong symptoms
    if not matched:
        return jsonify({
            "type": "message",
            "text": (
                "I couldn’t clearly recognize those symptoms.\n\n"
                "Please try simpler or common terms such as:\n"
                "• fever\n• headache\n• cough\n• vomiting\n• body pain"
            )
        })

    # Save matched symptoms
    session["symptoms"] = matched
    session["predictions"] = BUNDLE.predict_topk(matched)
    session["stage"] = "ASK_SYMPTOM_EXPLANATION"

    return jsonify({
        "type": "question",
        "text": (
            "Thank you for sharing.\n\n"
            "Would you like me to explain these symptoms "
            "in both general and medical terms?"
        ),
        "options": ["Yes", "No"]
    })

    # ---------------- CLARIFY ----------------
    if stage == "CLARIFY_SYMPTOMS":
        if msg.startswith("y"):
            session["symptoms"] = session["pending_symptoms"]
            session["predictions"] = BUNDLE.predict(session["symptoms"])
            session["stage"] = "ASK_SYMPTOM_EXPLANATION"
            return jsonify({
                "text": "Thank you for confirming. Shall I explain your symptoms?",
                "options": ["Yes", "No"]
            })
        else:
            session["stage"] = "ASK_SYMPTOMS"
            return jsonify({
                "text": "Okay, please rephrase or describe your symptoms again."
            })

    # ---------------- EXPLAIN SYMPTOMS ----------------
    if stage == "ASK_SYMPTOM_EXPLANATION":
        if msg.startswith("y"):
            session["stage"] = "SHOW_DISEASES"
            items = [
                f"{s}: {symptom_explanations.get(s, 'General bodily symptom.')}"
                for s in session["symptoms"]
            ]
            return jsonify({"items": items})
        else:
            session["stage"] = "SHOW_DISEASES"

    # ---------------- SHOW DISEASES ----------------
    if stage == "SHOW_DISEASES":
        session["stage"] = "FINAL"
        return jsonify({
            "text": "Based on your symptoms, these conditions are possible:",
            "items": [
                f"{p['condition']} ({p['probability']})"
                for p in session["predictions"]
            ],
            "advice": (
                "Please rest, stay hydrated, and monitor your symptoms."
            ),
            "disclaimer": (
                "⚠️ This is not a medical diagnosis. "
                "Seek professional care if symptoms worsen."
            )
        })

    return jsonify({"text": "I’m here whenever you’re ready."})

# ---------------------------
# Run
# ---------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)), debug=False)
