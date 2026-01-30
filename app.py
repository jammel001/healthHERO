import os
import re
from datetime import datetime

import joblib
import numpy as np
from rapidfuzz import process

from flask import (
    Flask,
    request,
    jsonify,
    render_template,
    session,
    send_file
)
from flask_cors import CORS
from flask_session import Session
from werkzeug.middleware.proxy_fix import ProxyFix

from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4

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
def extract_symptoms_from_text(text):
    text = text.lower()
    extracted = set()

    for phrase, symptom in SYMPTOM_ALIASES.items():
        if phrase in text:
            extracted.add(symptom)

    for symptom in CANONICAL_SYMPTOMS:
        if re.search(rf"\b{symptom}\b", text):
            extracted.add(symptom)

    return list(extracted)


# ---------------------------
# Model Bundle
# ---------------------------
class ModelBundle:
    def __init__(self):
        self.model = disease_model
        self.encoder = symptom_encoder
        self.label_encoder = label_encoder

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
# Severity & Advice
# ---------------------------
def determine_severity(days):
    if days <= 3:
        return "Mild"
    elif days <= 6:
        return "Moderate"
    return "Severe"


def emotional_advice(severity, name):
    if severity == "Mild":
        return f"{name}, this appears mild 🌱 Take rest and stay hydrated."
    if severity == "Moderate":
        return f"{name}, this needs attention 🤍 Please consult a doctor soon."
    return f"{name}, I’m concerned 🚨 Please seek urgent medical care."


# ---------------------------
# PDF Generator
# ---------------------------
def generate_prescription_pdf(data, filename):
    doc = SimpleDocTemplate(filename, pagesize=A4)
    styles = getSampleStyleSheet()
    content = []

    p = data["patient"]

    content.append(Paragraph("<b>HealthChero Medical Summary</b>", styles["Title"]))
    content.append(Paragraph(f"Name: {p['name']}", styles["Normal"]))
    content.append(Paragraph(f"Age: {p['age']}", styles["Normal"]))
    content.append(Paragraph(f"Gender: {p['gender']}", styles["Normal"]))
    content.append(Paragraph(f"Date: {datetime.now().strftime('%Y-%m-%d')}", styles["Normal"]))
    content.append(Paragraph("<b>Symptoms</b>", styles["Heading2"]))
    content.append(Paragraph(", ".join(data["symptoms"]), styles["Normal"]))
    content.append(Paragraph("<b>Severity</b>", styles["Heading2"]))
    content.append(Paragraph(data["severity"], styles["Normal"]))

    doc.build(content)


# ---------------------------
# Health Links
# ---------------------------
HEALTH_LINKS = {
    "Malaria": "https://www.who.int/news-room/fact-sheets/detail/malaria",
    "Typhoid": "https://www.cdc.gov/typhoid-fever",
    "Flu": "https://www.cdc.gov/flu"
}

# ---------------------------
# Routes
# ---------------------------
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/guidelines")
def guidelines():
    return render_template("guidelines.html")


@app.route("/api/diagnose", methods=["POST"])
def diagnose():
    data = request.json or {}
    user_input = (data.get("reply") or data.get("symptoms") or "").strip().lower()

    if "stage" not in session:
        session.clear()
        session["stage"] = "GREETING"
        session["patient"] = {}

    stage = session["stage"]

    if stage == "GREETING":
        session["stage"] = "ASK_CONSENT"
        return jsonify({"text": "Hello 👋 I’m HealthChero. Shall we continue?", "options": ["Yes", "No"]})

    if stage == "ASK_CONSENT":
        if user_input.startswith("y"):
            session["stage"] = "ASK_NAME"
            return jsonify({"text": "What is your name?"})
        session.clear()
        return jsonify({"text": "Take care 🙏"})

    if stage == "ASK_NAME":
        session["patient"]["name"] = user_input.title()
        session["stage"] = "ASK_AGE"
        return jsonify({"text": "How old are you?"})

    if stage == "ASK_AGE":
        if not user_input.isdigit():
            return jsonify({"text": "Enter a valid age."})
        session["patient"]["age"] = int(user_input)
        session["stage"] = "ASK_GENDER"
        return jsonify({"text": "Gender?", "options": ["Male", "Female", "Prefer not to say"]})

    if stage == "ASK_GENDER":
        session["patient"]["gender"] = user_input
        session["stage"] = "ASK_SYMPTOMS"
        return jsonify({"text": "Describe how you feel."})

    if stage == "ASK_SYMPTOMS":
        symptoms = extract_symptoms_from_text(user_input)
        if not symptoms:
            return jsonify({"text": "Please rephrase your symptoms."})
        session["symptoms"] = symptoms
        session["stage"] = "CONFIRM_PREDICTION"
        return jsonify({"text": "Shall I analyze possible conditions?", "options": ["Yes", "No"]})

    if stage == "CONFIRM_PREDICTION":
        if not user_input.startswith("y"):
            session.clear()
            return jsonify({"text": "Session ended."})

        predictions = BUNDLE.predict(session["symptoms"])
        session["predictions"] = predictions
        session["stage"] = "ASK_DURATION"

        items = [
            f"{p['condition']} ({int(p['probability']*100)}%)\n{p['description']}"
            for p in predictions
        ]

        return jsonify({"items": items, "text": "How many days have you felt this way?"})

    if stage == "ASK_DURATION":
        if not user_input.isdigit():
            return jsonify({"text": "Enter number of days (e.g. 3)."})

        days = int(user_input)
        severity = determine_severity(days)

        session["duration"] = days
        session["severity"] = severity
        session["stage"] = "FINAL"

        advice = emotional_advice(severity, session["patient"]["name"])
        links = [HEALTH_LINKS[p["condition"]] for p in session["predictions"] if p["condition"] in HEALTH_LINKS]

        return jsonify({
            "text": advice,
            "links": links,
            "options": ["Download Prescription"]
        })

    session.clear()
    return jsonify({"text": "Session finished."})


@app.route("/download")
def download():
    filename = "healthchero_prescription.pdf"
    generate_prescription_pdf(session, filename)
    return send_file(filename, as_attachment=True)


# ---------------------------
# Run
# ---------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)), debug=False)
