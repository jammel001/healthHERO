from flask import Flask, render_template, request, redirect, send_file
import pickle
import numpy as np
import difflib
from fpdf import FPDF
import os
import uuid

app = Flask(__name__)

# Load all necessary .pkl files
with open("disease_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("symptom_encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

with open("disease_to_description.pkl", "rb") as f:
    disease_desc = pickle.load(f)

with open("disease_to_precaution.pkl", "rb") as f:
    disease_precautions = pickle.load(f)

with open("symptom_to_explanation.pkl", "rb") as f:
    symptom_explanations = pickle.load(f)

all_symptoms = list(encoder.classes_)

# Severity advice system
def get_severity_advice(days):
    if days <= 3:
        return "🟢 Mild: Monitor your health, stay hydrated, and rest."
    elif 4 <= days <= 6:
        return "🟡 Moderate: Please consult a doctor if symptoms persist."
    else:
        return "🔴 Severe: Seek immediate medical attention!"

# Nearest symptom suggestion
def suggest_similar(symptom):
    matches = difflib.get_close_matches(symptom, all_symptoms, n=1, cutoff=0.6)
    return matches[0] if matches else None

# PDF Generator
def generate_pdf(patient_name, age, gender, symptoms, duration, predictions):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=14)
    pdf.cell(200, 10, txt="AI Health Diagnosis Report", ln=True, align="C")
    pdf.ln(10)

    pdf.cell(200, 10, txt=f"Patient: {patient_name}, Age: {age}, Gender: {gender}", ln=True)
    pdf.cell(200, 10, txt=f"Symptoms: {', '.join(symptoms)}", ln=True)
    pdf.cell(200, 10, txt=f"Duration: {duration} days", ln=True)
    pdf.cell(200, 10, txt=f"Severity: {get_severity_advice(duration)}", ln=True)
    pdf.ln(5)

    for disease in predictions:
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(200, 10, txt=f"Possible Disease: {disease}", ln=True)
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 8, f"Description: {disease_desc[disease]}")
        pdf.multi_cell(0, 8, f"Precautions: {', '.join(disease_precautions[disease])}")
        pdf.ln(5)

    file_path = f"static/{uuid.uuid4().hex}_report.pdf"
    pdf.output(file_path)
    return file_path

@app.route("/")
def home():
    return render_template("index.html", symptoms=all_symptoms)

@app.route("/diagnose", methods=["POST"])
def diagnose():
    name = request.form["name"]
    age = int(request.form["age"])
    gender = request.form["gender"]
    symptoms_raw = request.form["symptoms"].lower().split(",")
    symptoms = []

    explanations = []
    unresolved = []

    for s in symptoms_raw:
        s = s.strip()
        if s in symptom_explanations:
            symptoms.append(s)
            explanations.append((s, symptom_explanations[s]))
        else:
            suggestion = suggest_similar(s)
            if suggestion:
                symptoms.append(suggestion)
                explanations.append((suggestion, symptom_explanations.get(suggestion, "No explanation available.")))
            else:
                unresolved.append(s)

    if not symptoms:
        return f"None of the symptoms could be understood: {', '.join(unresolved)}"

    days = int(request.form["duration"])
    severity = get_severity_advice(days)

    # Prepare input vector
    input_vector = np.zeros(len(all_symptoms))
    for s in symptoms:
        if s in all_symptoms:
            idx = all_symptoms.index(s)
            input_vector[idx] = 1

    pred_probs = model.predict_proba([input_vector])[0]
    top_indices = np.argsort(pred_probs)[-5:][::-1]
    top_diseases = [model.classes_[i] for i in top_indices]

    description_list = [disease_desc[d] for d in top_diseases]
    precautions_list = [disease_precautions[d] for d in top_diseases]

    pdf_path = generate_pdf(name, age, gender, symptoms, days, top_diseases)

    return render_template(
        "result.html",
        name=name,
        age=age,
        gender=gender,
        symptoms=symptoms,
        explanations=explanations,
        duration=days,
        severity=severity,
        diseases=zip(top_diseases, description_list, precautions_list),
        pdf_path=pdf_path
    )

@app.route("/download/<path:filename>")
def download(filename):
    return send_file(filename, as_attachment=True)

@app.route("/restart")
def restart():
    return redirect("/")

if __name__ == "__main__":
    app.run(debug=True)
