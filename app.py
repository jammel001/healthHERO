from flask import Flask, render_template, request, send_file
import pickle
import numpy as np
from difflib import get_close_matches
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import os
import uuid

app = Flask(__name__)

# Load necessary files
with open("disease_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("symptom_encoder.pkl", "rb") as f:
    encoder = pickle.load(f)
with open("disease_to_description.pkl", "rb") as f:
    disease_descriptions = pickle.load(f)
with open("disease_to_precautions.pkl", "rb") as f:
    disease_precautions = pickle.load(f)
with open("symptom_to_explanation.pkl", "rb") as f:
    symptom_explanations = pickle.load(f)

all_symptoms = list(symptom_explanations.keys())

def get_severity_message(days):
    days = int(days)
    if days <= 3:
        return "Mild", "Stay hydrated and monitor your symptoms closely."
    elif 4 <= days <= 6:
        return "Moderate – Consult a doctor", "Consider seeing a doctor if symptoms persist."
    else:
        return "Severe – Urgent medical attention advised", "Seek immediate medical attention!"

def generate_pdf(name, age, gender, predictions, severity, tip):
    filename = f"{uuid.uuid4().hex}_prescription.pdf"
    filepath = os.path.join("static", filename)
    c = canvas.Canvas(filepath, pagesize=letter)
    width, height = letter

    c.setFont("Helvetica-Bold", 16)
    c.drawString(100, height - 50, "AI Health Diagnosis Prescription")

    c.setFont("Helvetica", 12)
    c.drawString(100, height - 100, f"Name: {name}")
    c.drawString(100, height - 120, f"Age: {age}")
    c.drawString(100, height - 140, f"Gender: {gender}")
    c.drawString(100, height - 160, f"Severity Level: {severity}")
    c.drawString(100, height - 180, f"Tip: {tip}")

    y = height - 220
    for i, disease in enumerate(predictions, 1):
        c.setFont("Helvetica-Bold", 13)
        c.drawString(100, y, f"{i}. {disease}")
        y -= 20
        c.setFont("Helvetica", 12)
        c.drawString(120, y, f"Description: {disease_descriptions.get(disease, 'N/A')}")
        y -= 20
        precautions = disease_precautions.get(disease, [])
        c.drawString(120, y, f"Precautions: {', '.join(precautions)}")
        y -= 30

    c.save()
    return filename

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/diagnose', methods=['POST'])
def diagnose():
    name = request.form['name']
    age = request.form['age']
    gender = request.form['gender']
    symptoms_raw = request.form['symptoms']
    duration = request.form['duration']

    symptoms = [s.strip().lower() for s in symptoms_raw.split(',') if s.strip()]
    confirmed_symptoms = []
    explanations = []
    suggestions = []

    for s in symptoms:
        if s in symptom_explanations:
            confirmed_symptoms.append(s)
            explanations.append(f"Symptom: {s.title()} – {symptom_explanations[s]}")
        else:
            close = get_close_matches(s, all_symptoms, n=1, cutoff=0.7)
            if close:
                suggestions.append(f"Did you mean '{close[0]}' instead of '{s}'?")
                confirmed_symptoms.append(close[0])
                explanations.append(f"Symptom: {close[0].title()} – {symptom_explanations[close[0]]}")
            else:
                explanations.append(f"Symptom: {s.title()} – No explanation found.")
    
    encoded_input = encoder.transform([' '.join(confirmed_symptoms)])
    predictions = model.predict_proba(encoded_input)[0]
    top_indices = predictions.argsort()[::-1][:5]
    top_diseases = [model.classes_[i] for i in top_indices]

    severity_level, tip = get_severity_message(duration)
    pdf_file = generate_pdf(name, age, gender, top_diseases, severity_level, tip)

    return render_template("result.html",
                           name=name,
                           age=age,
                           gender=gender,
                           explanations=explanations,
                           suggestions=suggestions,
                           diseases=top_diseases,
                           descriptions=[disease_descriptions[d] for d in top_diseases],
                           precautions=[disease_precautions[d] for d in top_diseases],
                           severity=severity_level,
                           tip=tip,
                           pdf_file=pdf_file)

@app.route('/download/<filename>')
def download(filename):
    return send_file(os.path.join('static', filename), as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)
