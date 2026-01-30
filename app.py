import os
import re
from datetime import datetime
from typing import List

import joblib
import numpy as np
from rapidfuzz import process

from flask import Flask, request, jsonify, render_template, session
from flask_cors import CORS
from flask_session import Session
from werkzeug.middleware.proxy_fix import ProxyFix

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
# Safe Loaders
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


# ---------------------------
# Load Models
# ---------------------------
symptom_encoder = safe_load(os.path.join(BASE_DIR, "symptom_encoder.pkl"))
symptom_embeddings = safe_numpy(os.path.join(BASE_DIR, "symptom_embeddings.npz"))
symptom_explanations = safe_load(
    os.path.join(BASE_DIR, "symptom_to_explanation.pkl")
) or {}

disease_model = safe_load(os.path.join(BASE_DIR, "disease_model.pkl"))
disease_descriptions = safe_load(
    os.path.join(BASE_DIR, "disease_to_description.pkl")
) or {}
disease_precautions = safe_load(
    os.path.join(BASE_DIR, "disease_to_precautions.pkl")
) or {}
label_encoder = safe_load(os.path.join(BASE_DIR, "label_encoder.pkl"))

# ---------------------------
# Symptom Canonical Map
# ---------------------------
CANONICAL_SYMPTOMS = [
    "fever", "headache", "vomiting", "nausea", "fatigue",
    "dizziness", "body pain", "loss of appetite", "cough",
    "shortness of breath", "chest pain", "diarrhea",
    "abdominal pain", "sore throat", "insomnia"
]

SYMPTOM_ALIASES = {
    "hot body": "fever",
    "head pain": "headache",
    "throwing up": "vomiting",
    "feeling nauseous": "nausea",
    "weak body": "fatigue",
    "no appetite": "loss of appetite",
    "dry cough": "cough",
    "tight chest": "chest pain",
    "difficulty breathing": "shortness of breath",
    "body aches": "body pain",
    "stomach pain": "abdominal pain",
    "running stomach": "diarrhea",
    "feeling dizzy": "dizziness",
    "throat pain": "sore throat",
    "cannot sleep": "insomnia",
}

# ---------------------------
# Symptom Extraction
# ---------------------------
def extract_symptoms_from_text(text: str):
    text = text.lower()
    extracted = set()
    clarifications = []

    for phrase, symptom in SYMPTOM_ALIASES.items():
        if phrase in text:
            extracted.add(symptom)

    for symptom in CANONICAL_SYMPTOMS:
        if re.search(rf"\b{re.escape(symptom)}\b", text):
            extracted.add(symptom)

    words = re.findall(r"[a-z]+", text)
    for word in words:
        match = process.extractOne(word, CANONICAL_SYMPTOMS, score_cutoff=85)
        if match and match[0] not in extracted:
            clarifications.append(
                f"Did you mean '{match[0]}' instead of '{word}'?"
            )
            extracted.add(match[0])

    return list(extracted), clarifications


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

    def match_symptoms(self, symptoms):
        matched = []
        for s in symptoms:
            if s in self.symptoms:
                matched.append(s)
            else:
                fuzzy = process.extractOne(s, self.symptoms, score_cutoff=85)
                if fuzzy:
                    matched.append(fuzzy[0])
        return list(set(matched))

    def predict(self, symptoms):
        if not symptoms or not self.model:
            return []

        X = self.encoder.transform([" ".join(symptoms)])
        probs = self.model.predict_proba(X)[0]
        top_idxs = np.argsort(probs)[::-1][:3]

        results = []
        for i in top_idxs:
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


@app.route("/api/diagnose", methods=["POST"])
def diagnose():
    data = request.json or {}
    user_input = (
        data.get("message")
        or data.get("symptoms")
        or data.get("reply")
        or ""
    ).strip().lower()

    if "stage" not in session:
        session.clear()
        session["stage"] = "GREETING"
        session["patient"] = {}

    stage = session["stage"]

    # ---------------- GREETING ----------------
    if stage == "GREETING":
        session["stage"] = "ASK_CONSENT"
        return jsonify({
            "text": "Hello 👋 I’m HealthChero. Shall we continue?",
            "options": ["Yes", "No"]
        })

    # ---------------- CONSENT ----------------
    if stage == "ASK_CONSENT":
        if user_input.startswith("y"):
            session["stage"] = "ASK_NAME"
            return jsonify({"text": "What is your name?"})

        session.clear()
        return jsonify({"text": "No problem. Take care."})

    # ---------------- NAME ----------------
    if stage == "ASK_NAME":
        session["patient"]["name"] = user_input.title()
        session["stage"] = "ASK_AGE"
        return jsonify({"text": "How old are you?"})

    # ---------------- AGE ----------------
    if stage == "ASK_AGE":
        if not user_input.isdigit():
            return jsonify({"text": "Please enter a valid age."})
        session["patient"]["age"] = int(user_input)
        session["stage"] = "ASK_GENDER"
        return jsonify({"text": "Gender?", "options": ["Male", "Female", "Prefer not to say"]})

    # ---------------- GENDER ----------------
    if stage == "ASK_GENDER":
        session["patient"]["gender"] = user_input
        session["stage"] = "ASK_SYMPTOMS"
        return jsonify({"text": "Describe how you feel."})

    # ---------------- SYMPTOMS ----------------
    if stage == "ASK_SYMPTOMS":
        matched, clarifications = extract_symptoms_from_text(user_input)

        if not matched:
            return jsonify({"text": "Please rephrase your symptoms."})

        session["symptoms"] = matched
        session["stage"] = "ASK_SYMPTOM_EXPLANATION"
        return jsonify({
            "text": "Do you want explanations of your symptoms?",
            "options": ["Yes", "No"]
        })

    # ---------------- EXPLANATIONS ----------------
    if stage == "ASK_SYMPTOM_EXPLANATION":
        if user_input.startswith("y"):
            explanations = [
                f"{s.title()}: {symptom_explanations.get(s, 'No explanation available.')}"
                for s in session["symptoms"]
            ]
            session["stage"] = "ASK_PREDICT_DISEASES"
            return jsonify({"items": explanations, "options": ["Continue"]})

        session["stage"] = "ASK_PREDICT_DISEASES"
        return jsonify({"text": "Proceeding to illness prediction."})

    # ---------------- PREDICT ----------------
    if stage == "ASK_PREDICT_DISEASES":
        predictions = BUNDLE.predict(session["symptoms"])
        session.clear()
        return jsonify({"items": predictions})

    session.clear()
    return jsonify({"text": "Session ended."})


# ---------------------------
# Run
# ---------------------------
if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 10000)),
        debug=False
    )
