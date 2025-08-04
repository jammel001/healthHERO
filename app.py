from flask import Flask, render_template, request, send_file
import joblib
import numpy as np
import random
import os
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

app = Flask(__name__)

# Load all necessary files
model = joblib.load("disease_model.pkl")
symptom_encoder = joblib.load("symptom_encoder.pkl")
disease_to_description = joblib.load("disease_to_description.pkl")
disease_to_precaution = joblib.load("disease_to_precautions.pkl")
symptom_to_explanation = joblib.load("symptom_to_explanation.pkl")

# Severity tips
severity_tips = {
    "Mild": "This seems to be a mild condition. Rest, hydrate, and monitor your symptoms.",
    "Moderate – Consult a doctor": "You may need to consult a doctor for proper examination and medication.",
    "Severe – Urgent medical attention advised": "Seek immediate medical attention. Do not delay visiting a hospital or clinic."
}

# Homepage route
@app.route('/')
def index():
    return render_template('index.html')

# Result route
@app.route('/result', methods=['POST'])
def result():
    name = request.form['name']
    age = request.form['age']
    gender = request.form['gender']
    raw_symptoms = request.form['symptoms']
    duration = int(request.form['duration'])

    # Process symptoms
    reported_symptoms = [s.strip().lower() for s in raw_symptoms.split(',') if s.strip()]
    explained_symptoms = []
    unknown_symptoms = []

    valid_symptoms = list(symptom_encoder.classes_)

    for sym in reported_symptoms:
        if sym in symptom_to_explanation:
            explained_symptoms.append((sym, symptom_to_explanation[sym]))
        else:
            # Fallback: suggest a nearby match
            similar = [vs for vs in valid_symptoms if sym[:3] in vs][:1]
            if similar:
                unknown_symptoms.append((sym, similar[0]))
            else:
                unknown_symptoms.append((sym, None))

    # Filter only known symptoms for prediction
    valid_input = [s for s in reported_symptoms if s in valid_symptoms]
    encoded = symptom_encoder.transform(valid_input)
    input_vector = np.zeros(len(symptom_encoder.classes_))
    input_vector[encoded] = 1

    # Predict
    prediction = model.predict_proba([input_vector])[0]
    top_indices = prediction.argsort()[::-1][:5]
    top_diseases = [(model.classes_[i], round(prediction[i]*100, 2)) for i in top_indices]

    # Add description and precautions
    detailed_diseases = []
    for disease, score in top_diseases:
        desc = disease_to_description.get(disease, "No description available.")
        precautions = disease_to_precaution.get(disease, ["No precautions listed."])
        detailed_diseases.append({
            "name": disease,
            "confidence": score,
            "description": desc,
            "precautions": precautions
        })

    # Severity
    if duration <= 3:
        severity = "Mild"
    elif 4 <= duration <= 6:
        severity = "Moderate – Consult a doctor"
    else:
        severity = "Severe – Urgent medical attention advised"

    tip = severity_tips[severity]

    return render_template("result.html", name=name, age=age, gender=gender,
                           symptoms=reported_symptoms,
                           explained_symptoms=explained_symptoms,
                           unknown_symptoms=unknown_symptoms,
                           diseases=detailed_diseases,
                           severity=severity, tip=tip)

# PDF download
@app.route('/download', methods=['POST'])
def download():
    name = request.form['name']
    age = request.form['age']
    gender = request.form['gender']
    diseases = request.form.getlist('diseases')
    descs = request.form.getlist('descs')
    tips = request.form.getlist('tips')

    file_path = f"{name}_diagnosis_{datetime.now().strftime('%Y%m%d%H%M%S')}.pdf"
    doc = SimpleDocTemplate(file_path, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph(f"Patient: {name}, Age: {age}, Gender: {gender}", styles["Title"]))
    story.append(Spacer(1, 12))

    for i, disease in enumerate(diseases):
        story.append(Paragraph(f"{i+1}. {disease}", styles["Heading2"]))
        story.append(Paragraph(descs[i], styles["BodyText"]))
        story.append(Paragraph("Precautions:", styles["Heading3"]))
        for p in tips[i].split(','):
            story.append(Paragraph(f"- {p.strip()}", styles["Normal"]))
        story.append(Spacer(1, 12))

    doc.build(story)
    return send_file(file_path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
