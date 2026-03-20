"""
HealthHero - Final Production Version
Includes:
- Conversational AI flow
- Symptom extraction
- Severity analysis
- ML prediction with confidence
- PDF report generation
- Hospital locator
"""

import os
import re
import io
from datetime import datetime
from flask import Flask, request, jsonify, render_template, send_file, session
from flask_session import Session
import joblib
import numpy as np
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

app = Flask(__name__)
app.secret_key = "supersecretkey"

# Session config
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

# =========================
# LOAD MODELS
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, "disease_model.pkl"))
encoder = joblib.load(os.path.join(BASE_DIR, "symptom_encoder.pkl"))
label_encoder = joblib.load(os.path.join(BASE_DIR, "label_encoder.pkl"))

disease_desc = joblib.load(os.path.join(BASE_DIR, "disease_to_description.pkl"))
disease_prec = joblib.load(os.path.join(BASE_DIR, "disease_to_precautions.pkl"))

# =========================
# SYMPTOMS
# =========================
SYMPTOMS = encoder.get_feature_names_out()

ALIASES = {
    "hot body": "fever",
    "tired": "fatigue",
    "weak": "fatigue",
    "vomit": "vomiting",
    "head pain": "headache",
    "stomach pain": "abdominal pain",
    "breathing problem": "shortness of breath"
}

# =========================
# HELPERS
# =========================
def extract_symptoms(text):
    text = text.lower().replace("i have", "").replace("i feel", "")
    found = []

    for s in SYMPTOMS:
        if s in text:
            found.append(s)

    for k, v in ALIASES.items():
        if k in text and v not in found:
            found.append(v)

    return list(set(found))


def calculate_severity(days):
    if days <= 3:
        return "Mild", "Rest and monitor symptoms."
    elif days <= 6:
        return "Moderate", "Consult a doctor if symptoms persist."
    else:
        return "Severe", "Seek medical attention immediately."


def predict(symptoms):
    X = encoder.transform([" ".join(symptoms)])
    probs = model.predict_proba(X)[0]

    top = np.argsort(probs)[::-1][:3]

    results = []
    for i, idx in enumerate(top, 1):
        prob = probs[idx]
        if prob < 0.05:
            continue

        disease = label_encoder.inverse_transform([idx])[0]

        results.append({
            "rank": i,
            "condition": disease,
            "probability": f"{int(prob*100)}%",
            "confidence": "High" if prob > 0.7 else "Medium" if prob > 0.4 else "Low",
            "description": disease_desc.get(disease, ""),
            "precautions": disease_prec.get(disease, [])
        })

    return results


def hospital_link():
    return "https://www.google.com/maps/search/hospitals+near+me"


# =========================
# ROUTES
# =========================
@app.route("/")
def home():
    return render_template("index.html")


# =========================
# CHATBOT
# =========================
@app.route("/api/diagnose", methods=["POST"])
def diagnose():

    data = request.json
    user_input = (data.get("reply") or data.get("symptoms") or "").strip()

    if "stage" not in session:
        session["stage"] = "start"

    stage = session["stage"]

    # ================= GREETING =================
    if stage == "start":
        session["stage"] = "ask_name"
        return jsonify({
            "text": "Hello 👋 I am HealthHero.\nWhat is your name?"
        })

    # ================= NAME =================
    if stage == "ask_name":
        session["name"] = user_input
        session["stage"] = "ask_age"
        return jsonify({"text": "How old are you?"})

    # ================= AGE =================
    if stage == "ask_age":
        match = re.findall(r"\d+", user_input)

        if not match:
            return jsonify({"text": "Please enter a valid age (e.g., 25)"})

        age = int(match[0])

        if age < 1 or age > 120:
            return jsonify({"text": "Enter a realistic age (1–120)"})

        session["age"] = age
        session["stage"] = "ask_gender"

        return jsonify({
            "text": "Gender?",
            "options": ["Male", "Female"]
        })

    # ================= GENDER =================
    if stage == "ask_gender":
        session["gender"] = user_input
        session["stage"] = "ask_symptoms"
        return jsonify({"text": "Describe your symptoms"})

    # ================= SYMPTOMS =================
    if stage == "ask_symptoms":
        symptoms = extract_symptoms(user_input)

        if not symptoms:
            return jsonify({
                "text": "I couldn't detect symptoms. Try 'fever', 'headache', etc.",
                "options": ["fever", "cough", "headache"]
            })

        session["symptoms"] = symptoms
        session["stage"] = "ask_duration"

        return jsonify({
            "text": f"Detected: {', '.join(symptoms)}\nHow many days?"
        })

    # ================= DURATION =================
    if stage == "ask_duration":
        match = re.findall(r"\d+", user_input)

        if not match:
            return jsonify({"text": "Please enter number of days (e.g., 3 days)"})

        days = int(match[0])

        severity, advice = calculate_severity(days)

        session["severity"] = severity
        session["stage"] = "predict"

        return jsonify({
            "text": f"Severity: {severity}\n{advice}\nAnalyze?",
            "options": ["Yes", "No"]
        })

    # ================= PREDICT =================
    if stage == "predict":

        if user_input.lower().startswith("n"):
            session.clear()
            return jsonify({"text": "Okay. Stay safe 🙏"})

        preds = predict(session["symptoms"])

        result = {
            "name": session.get("name"),
            "age": session.get("age"),
            "gender": session.get("gender"),
            "symptoms": session.get("symptoms"),
            "severity": session.get("severity"),
            "predictions": preds,
            "hospital_link": hospital_link()
        }

        session.clear()

        return jsonify({
            "text": "Here are possible conditions:",
            "severity": result["severity"],
            "predictions": preds,
            "hospital_link": result["hospital_link"],
            "disclaimer": "⚠️ Not a medical diagnosis."
        })


# =========================
# PDF DOWNLOAD
# =========================
@app.route("/api/download_pdf", methods=["POST"])
def download_pdf():

    data = request.json

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer)
    styles = getSampleStyleSheet()

    elements = []

    elements.append(Paragraph("HealthHero Medical Report", styles['Title']))
    elements.append(Spacer(1, 10))

    elements.append(Paragraph(f"Name: {data.get('name')}", styles['Normal']))
    elements.append(Paragraph(f"Age: {data.get('age')}", styles['Normal']))
    elements.append(Paragraph(f"Gender: {data.get('gender')}", styles['Normal']))
    elements.append(Spacer(1, 10))

    elements.append(Paragraph(f"Symptoms: {', '.join(data.get('symptoms',[]))}", styles['Normal']))
    elements.append(Paragraph(f"Severity: {data.get('severity')}", styles['Normal']))
    elements.append(Spacer(1, 10))

    elements.append(Paragraph("Predictions:", styles['Heading2']))

    for p in data.get("predictions", []):
        elements.append(Paragraph(
            f"{p['condition']} - {p['probability']} ({p['confidence']})",
            styles['Normal']
        ))

    elements.append(Spacer(1, 10))
    elements.append(Paragraph("⚠️ This is NOT a medical diagnosis.", styles['Normal']))

    doc.build(elements)
    buffer.seek(0)

    return send_file(buffer, as_attachment=True, download_name="report.pdf", mimetype="application/pdf")

@app.errorhandler(Exception)
def handle_exception(e):
    print("ERROR:", str(e))
    return jsonify({
        "text": "⚠️ Something went wrong. Please try again."
    }), 500
# =========================
# RUN
# =========================
if __name__ == "__main__":
    app.run(debug=True)
