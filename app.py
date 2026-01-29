import os
import io
from datetime import datetime
from typing import List
import re 
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
# ---------------------------
# Phase 2 Symptom Phrase Mapping
# ---------------------------

CANONICAL_SYMPTOMS = [
    "fever", "headache", "vomiting", "nausea",
    "fatigue", "dizziness", "body pain",
    "loss of appetite", "cough", "shortness of breath",
    "chest pain", "diarrhea", "abdominal pain"
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

    # 1️⃣ Phrase matching (multi-word phrases)
    for phrase, symptom in SYMPTOM_ALIASES.items():
        if phrase in text:
            extracted.add(symptom)

    # 2️⃣ Direct canonical symptom matching
    for symptom in CANONICAL_SYMPTOMS:
        pattern = r"\b" + re.escape(symptom) + r"\b"
        if re.search(pattern, text):
            extracted.add(symptom)

    # 3️⃣ Fuzzy matching (single words fallback)
    words = re.findall(r"[a-z]+", text)
    for word in words:
        match = process.extractOne(
            word,
            CANONICAL_SYMPTOMS,
            score_cutoff=85
        )
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

        if symptom_embeddings is not None:
            self.embedding_matrix = symptom_embeddings[list(symptom_embeddings.files)[0]]
        else:
            self.embedding_matrix = None

    def match_symptoms(self, symptoms):
    matched = []

    for s in symptoms:
        if s in self.symptoms:
            matched.append(s)
            continue

        fuzzy = process.extractOne(s, self.symptoms, score_cutoff=85)
        if fuzzy:
            matched.append(fuzzy[0])

    return list(set(matched))
    
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
        if user_input.startswith("y"):
            session["stage"] = "ASK_NAME"
            return jsonify({"text": "Great 👍 What is your name?"})
        else:
            session.clear()
            return jsonify({
                "text": "No problem. If you need help later, I’m here."
            })

    # -------------------------------
    # ASK NAME
    # -------------------------------
    if stage == "ASK_NAME":
        if not user_input:
            return jsonify({"text": "Please tell me your name."})

        session["patient"]["name"] = user_input.title()
        session["stage"] = "ASK_AGE"
        return jsonify({
            "text": f"Nice to meet you, {session['patient']['name']}.\n\nHow old are you?"
        })

    # -------------------------------
    # ASK AGE
    # -------------------------------
    if stage == "ASK_AGE":
        if not user_input.isdigit():
            return jsonify({
                "text": "Please enter your age in numbers (e.g., 25)."
            })

        age = int(user_input)
        if age < 0 or age > 120:
            return jsonify({"text": "Please enter a valid age."})

        session["patient"]["age"] = age
        session["stage"] = "ASK_GENDER"
        return jsonify({
            "text": "What is your gender?",
            "options": ["Male", "Female", "Prefer not to say"]
        })

    # -------------------------------
    # ASK GENDER
    # -------------------------------
    if stage == "ASK_GENDER":
        normalized = user_input.lower()
if normalized not in ["male", "female", "prefer not to say"]:
            return jsonify({
                "text": "Please choose one option.",
                "options": ["Male", "Female", "Prefer not to say"]
            })

        session["patient"]["gender"] = normalized
        session["stage"] = "ASK_SYMPTOMS"

        return jsonify({
            "text": (
                f"Thank you, {session['patient']['name']}.\n\n"
                "How are you feeling today? "
                "Please describe your symptoms in your own words."
            )
        })

  
    # -------------------------------
    # ASK SYMPTOMS
    # -------------------------------
  if stage == "ASK_SYMPTOMS":
    matched, clarifications = extract_symptoms_from_text(user_input)

    if not matched:
        return jsonify({
            "text": (
                "I couldn’t clearly identify your symptoms.\n\n"
                "Please describe how you feel in your own words.\n"
                "Example:\n"
                "• I feel very weak and my head is aching\n"
                "• I vomited and my body is hot"
            )
        })

    if clarifications:
        session["pending_symptoms"] = matched
        session["stage"] = "CLARIFY_SYMPTOMS"
        return jsonify({
            "text": "I want to be sure I understand you correctly:",
            "items": clarifications,
            "options": ["Yes", "No"]
        })

    session["symptoms"] = matched
    session["predictions"] = BUNDLE.predict(matched)
    session["stage"] = "ASK_SYMPTOM_EXPLANATION"

    return jsonify({
        "text": (
            "Thank you for explaining how you feel.\n\n"
            "Would you like me to explain each symptom "
            "in simple and medical terms?"
        ),
        "options": ["Yes", "No"]
    })

    # -------------------------------
    # CLARIFY SYMPTOMS
    # -------------------------------
    if stage == "CLARIFY_SYMPTOMS":
        if user_input.startswith("y"):
            session["symptoms"] = session.pop("pending_symptoms", [])
            session["predictions"] = BUNDLE.predict(session["symptoms"])
            session["stage"] = "ASK_SYMPTOM_EXPLANATION"
            return jsonify({
                "text": "Thank you for confirming. Would you like explanations?",
                "options": ["Yes", "No"]
            })
        else:
            session["stage"] = "ASK_SYMPTOMS"
            return jsonify({
                "text": "Okay, please rephrase or describe your symptoms again."
            })

    # -------------------------------
    # ASK SYMPTOM EXPLANATION
    # -------------------------------
    if stage == "ASK_SYMPTOM_EXPLANATION":
        if user_input.startswith("y"):
            session["stage"] = "ASK_PREDICT_DISEASES"
            return jsonify({
                "text": "Here is a clear explanation of each symptom:",
                "items": [
                    f"{s}: {symptom_explanations.get(s, 'General bodily symptom.')}"
                    for s in session["symptoms"]
                ],
                "options": ["Continue to possible conditions", "Stop here"]
            })
        else:
            session["stage"] = "ASK_PREDICT_DISEASES"
            return jsonify({
                "text": "Alright. Would you like me to predict the most likely conditions?",
                "options": ["Yes", "No"]
            })

    # -------------------------------
    # SHOW DISEASES
    # -------------------------------
    if stage == "ASK_PREDICT_DISEASES":
        if user_input.startswith("y") or "continue" in user_input:
            session["stage"] = "FINAL"
            return jsonify({
                "text": "Based on your symptoms, these conditions are possible:",
                "items": [
                    f"{p['condition']} ({p['probability']})"
                    for p in session.get("predictions", [])
                ],
                "advice": (
                    "Please rest, stay hydrated, and monitor your symptoms."
                ),
                "disclaimer": (
                    "⚠️ This is not a medical diagnosis. "
                    "Seek professional care if symptoms worsen."
                ),
                "options": ["Check another condition", "End session"]
            })
        else:
            session.clear()
            return jsonify({
                "text": "Okay. If you need help later, I’m always here"
            })

    # -------------------------------
    # FINAL / RESTART
    # -------------------------------
    if stage == "FINAL":
        session.clear()
        session["stage"] = "GREETING"
        return jsonify({
            "text": "Would you like to check another condition?",
            "options": ["Yes", "No"]
        })

    return jsonify({"text": "I’m here whenever you’re ready."})



# ---------------------------
# Run
# ---------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)), debug=False)
