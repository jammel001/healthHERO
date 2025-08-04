from flask import Flask, render_template, request, send_file
import pickle
import pandas as pd
import random
from fpdf import FPDF
import os

app = Flask(__name__)

# Load model and encoders
model = pickle.load(open('disease_model.pkl', 'rb'))
encoder = pickle.load(open('symptom_encoder.pkl', 'rb'))
descriptions = pd.read_csv("disease_description.csv", header=None, names=["disease", "description"])
precautions = pd.read_csv("disease_precaution.csv")

# Helper functions
def get_description(disease):
    row = descriptions[descriptions['disease'].str.lower() == disease.lower()]
    return row['description'].values[0] if not row.empty else "No description available."

def get_precautions(disease):
    row = precautions[precautions['Disease'].str.lower() == disease.lower()]
    if not row.empty:
        return [row[f'precaution_{i}'].values[0] for i in range(1, 5) if pd.notna(row[f'precaution_{i}'].values[0])]
    return ["No specific precautions available."]

def explain_symptom(symptom):
    explanations = {
        "headache": "Pain in the region of the head.",
        "cough": "A reflex to clear the throat or airway.",
        "fever": "Body temperature above normal due to infection.",
        "nausea": "Feeling of sickness with an urge to vomit.",
        "fatigue": "Extreme tiredness resulting from illness or exertion.",
    }
    return explanations.get(symptom.lower(), None)

def find_closest_symptom(symptom):
    all_symptoms = list(encoder.classes_)
    matches = [s for s in all_symptoms if symptom.lower() in s.lower()]
    return matches[:2] if matches else []

def predict_diseases(symptoms):
    encoded = encoder.transform(symptoms)
    input_df = pd.DataFrame([encoded], columns=encoder.get_feature_names_out())
    probabilities = model.predict_proba(input_df)[0]
    top_indices = probabilities.argsort()[::-1][:5]
    return [(model.classes_[i], round(probabilities[i]*100, 2)) for i in top_indices]

def estimate_severity(days):
    if days <= 3:
        return "Mild", "💡 Tip: Stay hydrated and take plenty of rest."
    elif days <= 6:
        return "Moderate", "⚠️ Tip: Monitor your symptoms and consult a healthcare provider."
    else:
        return "Severe", "🚨 Tip: Seek immediate medical attention from the nearest clinic or hospital."

def generate_pdf(name, results, severity, tip):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"AI Diagnosis Report for {name}", ln=1, align='C')
    pdf.cell(200, 10, txt=f"Severity Level: {severity}", ln=2)
    pdf.multi_cell(0, 10, f"Health Tip: {tip}")
    pdf.ln(5)
    for i, (disease, prob) in enumerate(results, 1):
        desc = get_description(disease)
        precs = get_precautions(disease)
        pdf.multi_cell(0, 10, f"{i}. {disease} ({prob}%)\nDescription: {desc}\nPrecautions: {', '.join(precs)}\n")
    path = f"static/{name}_report.pdf"
    pdf.output(path)
    return path

# Sequential conversation state
user_data = {}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/diagnose', methods=['POST'])
def diagnose():
    step = int(request.form.get('step', 1))
    if step == 1:
        user_data['name'] = request.form['name']
        return render_template('index.html', step=2, name=user_data['name'])

    elif step == 2:
        user_data['age'] = request.form['age']
        return render_template('index.html', step=3, name=user_data['name'])

    elif step == 3:
        user_data['gender'] = request.form['gender']
        return render_template('index.html', step=4, name=user_data['name'])

    elif step == 4:
        raw_symptoms = request.form['symptoms']
        symptoms = [s.strip().lower() for s in raw_symptoms.split(',')]
        user_data['symptoms'] = symptoms
        explanations = []
        suggestions = []
        for sym in symptoms:
            exp = explain_symptom(sym)
            if exp:
                explanations.append((sym, exp))
            else:
                suggestions.append((sym, find_closest_symptom(sym)))
        return render_template('index.html', step=5, name=user_data['name'], explanations=explanations, suggestions=suggestions)

    elif step == 5:
        return render_template('index.html', step=6, name=user_data['name'])

    elif step == 6:
        duration = int(request.form['duration'])
        user_data['duration'] = duration
        severity, tip = estimate_severity(duration)
        user_data['severity'] = severity
        user_data['tip'] = tip
        results = predict_diseases(user_data['symptoms'])
        user_data['results'] = results
        return render_template('result.html', name=user_data['name'], results=results, severity=severity, tip=tip)

    elif step == 7:
        pdf_path = generate_pdf(user_data['name'], user_data['results'], user_data['severity'], user_data['tip'])
        return send_file(pdf_path, as_attachment=True)

    elif step == 8:
        action = request.form['action']
        if action == 'yes':
            return render_template('index.html', step=1)
        else:
            message = "🌟 Wishing you good health! Stay positive and take care. – From Bara'u Magaji's AI Assistant"
            return render_template('result.html', final_message=message)

    return "Invalid step."

if __name__ == '__main__':
    app.run(debug=True)
