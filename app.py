import json
import pickle
import numpy as np
import logging
from flask import Flask, request, jsonify, render_template, session, send_file
from datetime import timedelta
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter
import os

app = Flask(__name__)
app.secret_key = "healthchero_secret"
app.permanent_session_lifetime = timedelta(minutes=30)

logging.basicConfig(level=logging.INFO)

# ------------------------------------------------
# LOAD MODELS
# ------------------------------------------------

with open("disease_model.pkl","rb") as f:
    disease_model = pickle.load(f)

with open("label_encoder.pkl","rb") as f:
    label_encoder = pickle.load(f)

with open("symptom_encoder.pkl","rb") as f:
    symptom_encoder = pickle.load(f)

with open("symptom_to_explanation.pkl","rb") as f:
    symptom_explanations = pickle.load(f)

with open("disease_to_description.pkl","rb") as f:
    disease_descriptions = pickle.load(f)

with open("disease_to_precautions.pkl","rb") as f:
    disease_precautions = pickle.load(f)

with open("model_tables.pkl","rb") as f:
    model_tables = pickle.load(f)

embedding_data = np.load("symptom_embeddings.npz", allow_pickle=True)
symptom_vectors = embedding_data["embeddings"]
symptom_names = embedding_data["symptoms"]

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# ------------------------------------------------
# HOSPITAL DATA
# ------------------------------------------------

try:
    with open("hospitals.json") as f:
        hospitals = json.load(f)
except:
    hospitals = {}

# ------------------------------------------------
# FUNCTIONS
# ------------------------------------------------

def extract_symptoms(text, threshold=0.55):

    query = embedding_model.encode([text])

    sims = cosine_similarity(query, symptom_vectors)[0]

    idx = np.argsort(sims)[::-1][:5]

    detected = []

    for i in idx:
        if sims[i] > threshold:
            detected.append(symptom_names[i])

    return detected


def explain_symptoms(symptoms):

    out = []

    for s in symptoms:
        if s in symptom_explanations:
            out.append(f"{s.title()}: {symptom_explanations[s]}")

    return out


def encode_symptoms(symptoms):

    return symptom_encoder.transform([symptoms])


def predict(symptoms):

    X = encode_symptoms(symptoms)

    probs = disease_model.predict_proba(X)[0]

    idx = np.argsort(probs)[::-1][:3]

    results = []

    for i in idx:

        disease = label_encoder.inverse_transform([i])[0]

        results.append({
            "condition": disease,
            "confidence": round(probs[i]*100,2),
            "description": disease_descriptions.get(disease,""),
            "precautions": disease_precautions.get(disease,[])
        })

    return results


def nearest_hospitals(city):

    if city and city.title() in hospitals:
        return hospitals[city.title()][:3]

    return []


def severity(days):

    try:
        d = int(days)

        if d <= 3:
            return "Mild"

        if d <= 6:
            return "Moderate"

        return "Severe"

    except:
        return "Unknown"

# ------------------------------------------------
# PDF GENERATION
# ------------------------------------------------

def generate_pdf(patient, symptoms, predictions, severity_level):

    file_path = "report.pdf"

    styles = getSampleStyleSheet()

    elements = []

    elements.append(Paragraph("HealthChero Medical Report", styles['Title']))
    elements.append(Spacer(1,20))

    elements.append(Paragraph(f"Patient Name: {patient['name']}", styles['Normal']))
    elements.append(Paragraph(f"Age: {patient['age']}", styles['Normal']))
    elements.append(Paragraph(f"Gender: {patient['gender']}", styles['Normal']))
    elements.append(Paragraph(f"Location: {patient['location']}", styles['Normal']))

    elements.append(Spacer(1,20))

    elements.append(Paragraph("Detected Symptoms:", styles['Heading2']))

    for s in symptoms:
        elements.append(Paragraph(s, styles['Normal']))

    elements.append(Spacer(1,20))

    elements.append(Paragraph(f"Severity Level: {severity_level}", styles['Heading2']))

    elements.append(Spacer(1,20))

    elements.append(Paragraph("Predicted Conditions:", styles['Heading2']))

    table_data = [["Condition","Confidence"]]

    for p in predictions:
        table_data.append([p["condition"], f"{p['confidence']}%"])

    table = Table(table_data)

    elements.append(table)

    doc = SimpleDocTemplate(file_path,pagesize=letter)

    doc.build(elements)

    return file_path

# ------------------------------------------------
# SESSION RESET
# ------------------------------------------------

def reset():

    session["stage"] = "ASK_NAME"

    session["patient"] = {
        "name":None,
        "age":None,
        "gender":None,
        "location":None,
        "symptoms":[]
    }

# ------------------------------------------------
# ROUTES
# ------------------------------------------------

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/download-report")
def download_report():

    path = "report.pdf"

    if os.path.exists(path):
        return send_file(path, as_attachment=True)

    return "Report not available"


@app.route("/api/diagnose", methods=["POST"])
def diagnose():

    if "stage" not in session:
        reset()

    stage = session["stage"]

    data = request.get_json()

    user_input = data.get("reply") or data.get("symptoms")

# --------------------------

    if stage == "ASK_NAME":

        session["patient"]["name"] = user_input
        session["stage"] = "ASK_AGE"

        return jsonify({"text":f"Nice to meet you {user_input}. What is your age?"})

# --------------------------

    if stage == "ASK_AGE":

        session["patient"]["age"] = user_input
        session["stage"] = "ASK_GENDER"

        return jsonify({"text":"What is your gender?"})

# --------------------------

    if stage == "ASK_GENDER":

        session["patient"]["gender"] = user_input
        session["stage"] = "ASK_LOCATION"

        return jsonify({"text":"Which city are you currently in?"})

# --------------------------

    if stage == "ASK_LOCATION":

        session["patient"]["location"] = user_input
        session["stage"] = "ASK_SYMPTOMS"

        return jsonify({"text":"Please describe your symptoms."})

# --------------------------

    if stage == "ASK_SYMPTOMS":

        symptoms = extract_symptoms(user_input)

        session["patient"]["symptoms"] = symptoms

        session["stage"] = "ASK_DURATION"

        return jsonify({
            "text":"Detected symptoms:",
            "items":explain_symptoms(symptoms)
        })

# --------------------------

    if stage == "ASK_DURATION":

        sev = severity(user_input)

        symptoms = session["patient"]["symptoms"]

        preds = predict(symptoms)

        hospitals_list = nearest_hospitals(session["patient"]["location"])

        generate_pdf(session["patient"], symptoms, preds, sev)

        session["stage"] = "RESULT"

        return jsonify({

            "text":"Possible conditions based on your symptoms:",

            "predictions":preds,

            "severity":sev,

            "hospitals":hospitals_list,

            "advice":"Consult a healthcare professional.",

            "report_url":"/download-report"

        })

# --------------------------

    return jsonify({"text":"Session finished. Restart to begin again."})

# ------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True)
