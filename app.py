from flask import Flask, render_template, request, send_file, redirect, url_for
import joblib
import difflib
import re
from fpdf import FPDF
import os

app = Flask(__name__)

# Load model and encoders
model = joblib.load("disease_model.pkl")
symptom_encoder = joblib.load("symptom_encoder.pkl")
disease_to_description = joblib.load("disease_to_description.pkl")
disease_to_precautions = joblib.load("disease_to_precautions.pkl")
symptom_to_explanation = joblib.load("symptom_to_explanation.pkl")

all_symptoms = symptom_encoder.classes_
symptom_synonyms = {
    'fever': 'fever', 'high temperature': 'fever', 'cold': 'chills', 'nausea': 'vomiting',
    'headach': 'headache', 'dizzy': 'dizziness', 'flu': 'sneezing', 'tired': 'fatigue'
}

@app.route("/", methods=["GET", "POST"])
def index():
    return render_template("index.html")

@app.route("/diagnose", methods=["POST"])
def diagnose():
    name = request.form["name"]
    age = request.form["age"]
    gender = request.form["gender"]
    location = request.form["location"]
    symptoms_raw = request.form["symptoms"].lower().split(",")
    duration_input = request.form["duration"]

    final_symptoms = []
    explanations = []

    for s in symptoms_raw:
        s = s.strip().lower()
        s = symptom_synonyms.get(s, s)
        if s in all_symptoms:
            final_symptoms.append(s)
            expl = symptom_to_explanation.get(s, {})
            explanations.append((s, expl.get("general", "No explanation available."), expl.get("medical", "No explanation available.")))
        else:
            suggestion = difflib.get_close_matches(s, all_symptoms, n=1, cutoff=0.6)
            if suggestion:
                final_symptoms.append(suggestion[0])
                expl = symptom_to_explanation.get(suggestion[0], {})
                explanations.append((suggestion[0], expl.get("general", "No explanation available."), expl.get("medical", "No explanation available.")))

    diseases = []
    if final_symptoms:
        X_input = symptom_encoder.transform([final_symptoms])
        probs = model.predict_proba(X_input)[0]
        top_indices = probs.argsort()[-5:][::-1]
        for i in top_indices:
            disease = model.classes_[i]
            prob = round(probs[i] * 100, 2)
            key = disease.lower().strip()
            desc = disease_to_description.get(key, "No description available.")
            precautions = disease_to_precautions.get(key, ["No precautions available."])
            diseases.append({"name": disease.title(), "probability": prob, "description": desc, "precautions": precautions})

    duration = int(re.search(r"\d+", duration_input).group()) if re.search(r"\d+", duration_input) else 0
    if duration <= 3:
        severity = "🟢 Mild – Monitor your symptoms and rest."
        tip = "Even mild symptoms matter. Hydrate, rest, and listen to your body."
    elif 4 <= duration <= 6:
        severity = "🟡 Moderate – Please consult a doctor soon."
        tip = "Early care prevents complications. Your health is a priority."
    else:
        severity = "🔴 Severe – Seek urgent medical attention!"
        tip = "Act now! A quick response can save your life."

    # Generate PDF
    pdf_path = generate_pdf(name, diseases, severity, tip)

    return render_template("result.html", name=name, location=location, explanations=explanations, diseases=diseases, severity=severity, tip=tip, pdf_path=pdf_path)

@app.route("/download")
def download():
    pdf_path = request.args.get("path")
    return send_file(pdf_path, as_attachment=True)

def generate_pdf(name, diseases, severity, tip):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(200, 10, f"Diagnosis Report for {name}", ln=True, align='C')
    pdf.set_font("Arial", size=12)
    pdf.ln(10)

    for d in diseases:
        pdf.multi_cell(0, 10, f"Disease: {d['name']} ({d['probability']}%)\nDescription: {d['description']}\nPrecautions: {', '.join(d['precautions'])}\n")
        pdf.ln(2)

    pdf.ln(5)
    pdf.multi_cell(0, 10, f"Severity: {severity}\nHealth Tip: {tip}")

    path = f"prescription_{name.lower().replace(' ', '_')}.pdf"
    pdf.output(path)
    return path

@app.route("/restart")
def restart():
    return redirect(url_for("index"))

@app.route("/end")
def end():
    return "<h2>🙏 Thank you for using this AI Health Assistant! Stay positive, stay healthy, and take care of yourself every day. 💖</h2>"

if __name__ == "__main__":
    app.run(debug=True)
