# ✅ File: app.py
from flask import Flask, render_template, request
import pandas as pd
import joblib
import re
from difflib import get_close_matches

app = Flask(__name__)

# Load model components
model = joblib.load("rf_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")
label_encoder = joblib.load("label_encoder.pkl")
df = pd.read_csv("merged_symptoms.csv", encoding="latin1")

# Prepare explanations
defined = df[['Symptoms.1', 'General Explanation', 'Medical Explanation']].dropna()
explained_symptoms = {
    row['Symptoms.1'].strip().lower(): {
        "general": row['General Explanation'],
        "medical": row['Medical Explanation']
    }
    for _, row in defined.iterrows()
}

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/diagnose", methods=["POST"])
def diagnose():
    name = request.form['name']
    age = request.form['age']
    gender = request.form['gender']
    location = request.form['location']
    duration = int(''.join(filter(str.isdigit, request.form['duration'])))
    symptoms_input = request.form['symptoms'].lower()
    symptoms = [s.strip() for s in symptoms_input.split(',') if s.strip()]

    explained = []
    confirmed_symptoms = []
    for s in symptoms:
        if s in explained_symptoms:
            explained.append((s, explained_symptoms[s]['general'], explained_symptoms[s]['medical']))
            confirmed_symptoms.append(s)
        else:
            suggestion = get_close_matches(s, explained_symptoms.keys(), n=1, cutoff=0.6)
            if suggestion:
                explained.append((s, f"Did you mean '{suggestion[0]}'?", "No exact match found."))
            else:
                explained.append((s, "Not found", "Not found"))

    confirm = request.form.get('confirm')
    if confirm != "yes":
        return render_template("result.html", name=name, canceled=True)

    text = ", ".join(confirmed_symptoms)
    X = vectorizer.transform([text])
    prob = model.predict_proba(X)[0]
    top5 = prob.argsort()[-5:][::-1]

    predictions = []
    for idx in top5:
        disease = label_encoder.inverse_transform([idx])[0]
        row = df[df['Disease'] == disease].iloc[0]
        desc = row['Description'] if pd.notna(row['Description']) else "No description"
        precautions = [row.get(f"Precaution_{i}", "") for i in range(1, 5) if pd.notna(row.get(f"Precaution_{i}"))]
        predictions.append({
            "disease": disease,
            "confidence": f"{prob[idx]*100:.2f}%",
            "desc": desc,
            "precautions": precautions
        })

    if duration <= 3:
        severity = "🟢 Mild"
    elif 4 <= duration <= 6:
        severity = "🟠 Moderate – You should consult a doctor."
    else:
        severity = "🔴 Severe – Urgent medical attention advised!"

    return render_template("result.html", name=name, predictions=predictions, location=location, severity=severity, again=True)

if __name__ == "__main__":
    app.run(debug=True)
