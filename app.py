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
symptom_explanations = safe_load(os.path.join(BASE_DIR, "symptom_to_explanation.pkl")) or {}

disease_model = safe_load(os.path.join(BASE_DIR, "disease_model.pkl"))
disease_descriptions = safe_load(os.path.join(BASE_DIR, "disease_to_description.pkl")) or {}
disease_precautions = safe_load(os.path.join(BASE_DIR, "disease_to_precautions.pkl")) or {}
label_encoder = safe_load(os.path.join(BASE_DIR, "label_encoder.pkl"))

# =====================================================
# Phase 2 — Symptom Phrase Mapping (✅ CORRECT LOCATION)
# =====================================================
CANONICAL_SYMPTOMS = [
    "fever", "headache", "vomiting", "nausea", "fatigue",
    "dizziness", "body pain", "loss of appetite", "cough",
    "shortness of breath", "chest pain", "diarrhea",
    "abdominal pain", "sore throat", "insomnia"
]

SYMPTOM_ALIASES = {    
# =======================
    # FEVER / TEMPERATURE
    # =======================
    "hot body": "fever",
    "body hot": "fever",
    "high temperature": "fever",
    "feeling hot": "fever",
    "body heat": "fever",
    "running temperature": "fever",
    "burning body": "fever",
    "hot inside": "fever",
    "i feel hot": "fever",
    "my body is hot": "fever",
    "feverish": "fever",
    "chills and fever": "fever",
    "cold and hot": "fever",
    # =======================
    # HEADACHE
    # =======================
    "head pain": "headache",
    "pain in my head": "headache",
    "pounding head": "headache",
    "heavy head": "headache",
    "head is aching": "headache",
    "my head hurts": "headache",
    "pain around my head": "headache",
    "migraine": "headache",
    "sharp head pain": "headache",
    "head pressure": "headache",
    "throbbing head": "headache",
    # =======================
    # VOMITING / NAUSEA
    # =======================
    "throwing up": "vomiting",
    "vomit": "vomiting",
    "vomited": "vomiting",
    "i vomited": "vomiting",
    "throw up": "vomiting",
    "throwing out": "vomiting",
    "retching": "vomiting",
    "nausea and vomiting": "vomiting",
    "feel like vomiting": "vomiting",
    "feel like throwing up": "vomiting",
    "stomach upset and vomiting": "vomiting",
    # =======================
    # NAUSEA ONLY
    # =======================
    "feeling nauseous": "nausea",
    "feeling sick": "nausea",
    "queasy stomach": "nausea",
    "unsettled stomach": "nausea",
    # =======================
    # FATIGUE / WEAKNESS
    # =======================
    "weak body": "fatigue",
    "very weak": "fatigue",
    "tired all the time": "fatigue",
    "no strength": "fatigue",
    "body weakness": "fatigue",
    "feeling weak": "fatigue",
    "always tired": "fatigue",
    "low energy": "fatigue",
    "exhausted": "fatigue",
    "easily tired": "fatigue",
    # =======================
    # APPETITE
    # =======================
    "loss of appetite": "loss of appetite",
    "no appetite": "loss of appetite",
    "poor appetite": "loss of appetite",
    "cannot eat": "loss of appetite",
    "not eating well": "loss of appetite",
    "food does not interest me": "loss of appetite",
    "reduced appetite": "loss of appetite",
    # =======================
    # COUGH
    # =======================
    "dry cough": "cough",
    "persistent cough": "cough",
    "continuous cough": "cough",
    "coughing a lot": "cough",
    "coughing": "cough",
    "night cough": "cough",
    # =======================
    # CHEST
    # =======================
    "chest pain": "chest pain",
    "pain in my chest": "chest pain",
    "tight chest": "chest pain",
    "chest tightness": "chest pain",
    "burning chest": "chest pain",
    # =======================
    # BREATHING
    # =======================
    "shortness of breath": "shortness of breath",
    "difficulty breathing": "shortness of breath",
    "hard to breathe": "shortness of breath",
    "breathing problem": "shortness of breath",
    "fast breathing": "shortness of breath",
    # =======================
    # BODY PAIN
    # =======================
    "body pain": "body pain",
    "general body pain": "body pain",
    "body aches": "body pain",
    "pain all over my body": "body pain",
    "muscle pain": "muscle pain",
    "joint pain": "joint pain",
    "bone pain": "joint pain",
    # =======================
    # ABDOMINAL / STOMACH
    # =======================
    "stomach pain": "abdominal pain",
    "stomach ache": "abdominal pain",
    "abdominal pain": "abdominal pain",
    "pain in my stomach": "abdominal pain",
    "belly pain": "abdominal pain",
    "lower stomach pain": "abdominal pain",
    "upper stomach pain": "abdominal pain",
    # =======================
    # DIARRHEA
    # =======================
    "running stomach": "diarrhea",
    "loose stool": "diarrhea",
    "watery stool": "diarrhea",
    "frequent stool": "diarrhea",
    "diarrhoea": "diarrhea",
    # =======================
    # DIZZINESS
    # =======================
    "dizziness": "dizziness",
    "feeling dizzy": "dizziness",
    "lightheaded": "dizziness",
    "head spinning": "dizziness",
    "about to faint": "dizziness",
    # =======================
    # SORE THROAT
    # =======================
    "sore throat": "sore throat",
    "throat pain": "sore throat",
    "pain when swallowing": "sore throat",
    "itchy throat": "sore throat",
    # =======================
    # SLEEP
    # =======================
    "cannot sleep": "insomnia",
    "difficulty sleeping": "insomnia",
    "poor sleep": "insomnia",
    "sleepless nights": "insomnia"
}

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
            clarifications.append(f"Did you mean '{match[0]}' instead of '{word}'?")
            extracted.add(match[0])

    return list(extracted), clarifications

# ===========================
# Model Bundle
# ===========================
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

    def predict(self, matched):
        if not matched or not self.model:
            return []

        X = self.encoder.transform([" ".join(matched)])
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

    # Initialize conversation
    if "stage" not in session:
        session.clear()
        session["stage"] = "GREETING"
        session["patient"] = {}

    stage = session["stage"]

    # -------------------------------
    # GREETING
    # -------------------------------
    if stage == "GREETING":
        session["stage"] = "ASK_CONSENT"
        return jsonify({
            "text": (
                "Hello 👋 I’m HealthChero, your virtual health assistant.\n\n"
                "I’ll ask a few questions to better understand how you feel.\n"
                "⚠️ This is not a medical diagnosis.\n\n"
                "Shall we continue?"
            ),
            "options": ["Yes", "No"]
        })

    # -------------------------------
    # ASK CONSENT
    # -------------------------------
    if stage == "ASK_CONSENT":
        if user_input in ["yes", "y", "ok", "okay", "continue"]:
            session["stage"] = "ASK_NAME"
            return jsonify({
                "text": "Great 👍 What is your name?"
            })

        if user_input in ["no", "n"]:
            session.clear()
            return jsonify({
                "text": "No problem. If you need help later, I’m here."
            })

        return jsonify({
            "text": "Please choose one option.",
            "options": ["Yes", "No"]
        })

    if stage == "ASK_NAME":
        session["patient"]["name"] = user_input.title()
        session["stage"] = "ASK_AGE"
        return jsonify({"text": "How old are you?"})

    if stage == "ASK_AGE":
        if not user_input.isdigit():
            return jsonify({"text": "Please enter a valid age."})
        session["patient"]["age"] = int(user_input)
        session["stage"] = "ASK_GENDER"
        return jsonify({"text": "What is your gender?", "options": ["Male", "Female", "Prefer not to say"]})

    if stage == "ASK_GENDER":
        session["patient"]["gender"] = user_input
        session["stage"] = "ASK_SYMPTOMS"
        return jsonify({"text": "Please describe how you are feeling."})

    if stage == "ASK_SYMPTOMS":
        matched, clarifications = extract_symptoms_from_text(user_input)

        if not matched:
            return jsonify({"text": "I couldn’t understand your symptoms. Please describe again."})

        if clarifications:
            session["pending_symptoms"] = matched
            session["stage"] = "CLARIFY_SYMPTOMS"
            return jsonify({"text": "Please confirm:", "items": clarifications, "options": ["Yes", "No"]})

       session["symptoms"] = matched
       session["stage"] = "ASK_SYMPTOM_EXPLANATION"

  if stage == "ASK_SYMPTOM_EXPLANATION":
    if user_input.startswith("y"):
        explanations = []
        for s in session["symptoms"]:
            explanations.append(
                f"🔹 {s.title()}: {symptom_explanations.get(s, 'No explanation available.')}"
            )

        session["stage"] = "ASK_PREDICT_DISEASES"
        return jsonify({
            "text": "Here’s an explanation of your symptoms:",
            "items": explanations,
            "options": ["Continue to illness prediction", "Stop"]
        })
    else:
        session["stage"] = "ASK_PREDICT_DISEASES"
        return jsonify({
            "text": "Okay. Shall I predict the most likely illnesses?",
            "options": ["Yes", "No"]
        })
  if stage == "ASK_PREDICT_DISEASES":
    if user_input.startswith("y"):
        session["predictions"] = BUNDLE.predict(session["symptoms"])
        session["stage"] = "ASK_ILLNESS_EXPLANATION"

        return jsonify({
            "text": "Based on your symptoms, these conditions are most likely:",
            "items": session["predictions"],
            "options": ["Explain these illnesses", "End session"]
        })
    else:
        session.clear()
        return jsonify({"text": "Alright. Take care 🙏"})
 if stage == "ASK_ILLNESS_EXPLANATION":
    if user_input.startswith("y"):
        details = []
        for p in session["predictions"]:
            details.append(
                f"🩺 {p['condition']}:\n{p['description']}\nPrecautions: {', '.join(p['precautions'])}"
            )

        session["stage"] = "FINAL"
        return jsonify({
            "text": "Here are details about the possible illnesses:",
            "items": details,
            "disclaimer": "⚠️ This is not a medical diagnosis."
        })
    else:
        session.clear()
        return jsonify({"text": "Thank you for using HealthChero."})

    if stage == "CLARIFY_SYMPTOMS":
        if user_input.startswith("y"):
            session["symptoms"] = session.pop("pending_symptoms", [])
            session["predictions"] = BUNDLE.predict(session["symptoms"])
            session["stage"] = "FINAL"
            return jsonify({"items": session["predictions"]})
        session["stage"] = "ASK_SYMPTOMS"
        return jsonify({"text": "Okay, please rephrase your symptoms."})

    session.clear()
    return jsonify({"text": "Session ended. Refresh to start again."})

# ---------------------------
# Run
# ---------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)), debug=False)
