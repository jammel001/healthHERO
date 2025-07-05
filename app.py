from flask import Flask, render_template, request, redirect, url_for
import joblib
import json
import numpy as np

app = Flask(__name__)

# Load trained model and data
model = joblib.load("disease_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")
with open("symptom_list.json") as f:
    symptom_list = json.load(f)
with open("disease_info.json") as f:
    disease_info = json.load(f)

# Temporary in-memory storage for conversation steps
patient_data = {}

@app.route("/", methods=["GET", "POST"])
def index():
    step = request.args.get("step", "1")

    if request.method == "POST":
        if step == "1":
            patient_data["name"] = request.form["name"]
            return redirect(url_for("index", step="2"))
        elif step == "2":
            patient_data["age"] = request.form["age"]
            return redirect(url_for("index", step="3"))
        elif step == "3":
            patient_data["gender"] = request.form["gender"]
            return redirect(url_for("index", step="4"))
        elif step == "4":
            patient_data["location"] = request.form["location"]
            return redirect(url_for("index", step="5"))
        elif step == "5":
            patient_data["symptoms"] = request.form["symptoms"]
            return redirect(url_for("index", step="6"))
        elif step == "6":
            patient_data["days"] = int(request.form["days"])
            return redirect(url_for("result"))

    return render_template("index.html", step=step, data=patient_data)


@app.route("/result")
def result():
    # Process input
    symptoms_input = patient_data["symptoms"].lower()
    cleaned_input = [s.strip().replace("-", "").replace(".", "") for s in symptoms_input.split(",")]
    input_vector = np.array([[1 if s in cleaned_input else 0 for s in symptom_list]])

    # Predict top 5 diseases
    probs = model.predict_proba(input_vector)[0]
    top_indices = probs.argsort()[-5:][::-1]

    results = []
    for idx in top_indices:
        disease = label_encoder.inverse_transform([idx])[0]
        confidence = round(probs[idx] * 100, 2)
        info = disease_info.get(disease, {})
        results.append({
            "name": disease.title(),
            "confidence": confidence,
            "description": info.get("description", "No description available."),
            "precautions": info.get("precautions", [])
        })
from flask import send_file
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
import io

@app.route("/download")
def download():
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    y = height - 50
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, y, "🧾 AI Health Assistant Diagnosis Report")
    y -= 30

    c.setFont("Helvetica", 12)
    c.drawString(50, y, f"👤 Name: {patient_data['name']}")
    y -= 20
    c.drawString(50, y, f"🎂 Age: {patient_data['age']} | ⚧ Gender: {patient_data['gender']}")
    y -= 20
    c.drawString(50, y, f"📍 Location: {patient_data['location']}")
    y -= 20
    c.drawString(50, y, f"💬 Symptoms: {patient_data['symptoms']}")
    y -= 30

    # Severity
    days = patient_data['days']
    if days <= 3:
        severity = "Moderate – Rest and monitor closely. Visit a doctor if worsened."
    elif days <= 6:
        severity = "Mild to Severe – Consider medical consultation soon."
    else:
        severity = "Severe – Visit the nearest hospital immediately."

    c.drawString(50, y, f"⏳ Duration of illness: {days} days")
    y -= 20
    c.drawString(50, y, f"⚠️ Severity: {severity}")
    y -= 30

    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Predicted Diseases:")
    y -= 20

    # Include top 5 diseases
    probs = model.predict_proba(np.array([[1 if s in patient_data['symptoms'].lower() else 0 for s in symptom_list]]))[0]
    top_indices = probs.argsort()[-5:][::-1]

    for idx in top_indices:
        disease = label_encoder.inverse_transform([idx])[0]
        info = disease_info.get(disease, {})
        confidence = round(probs[idx] * 100, 2)
        y -= 15
        c.setFont("Helvetica-Bold", 11)
        c.drawString(60, y, f"{disease.title()} ({confidence}%)")
        y -= 15
        c.setFont("Helvetica", 10)
        c.drawString(60, y, "Description: " + info.get("description", "N/A"))
        y -= 15
        c.drawString(60, y, "Precautions:")
        for p in info.get("precautions", []):
            y -= 12
            c.drawString(75, y, f"- {p}")
        y -= 20
        if y < 100:
            c.showPage()
            y = height - 50

    c.drawString(50, y, "📄 This is a computer-generated health suggestion. Please consult a certified physician.")
    c.save()

    buffer.seek(0)
    return send_file(buffer, as_attachment=True, download_name="diagnosis_report.pdf", mimetype="application/pdf")

    # Severity check based on days
    days = patient_data["days"]
    if days <= 3:
        severity = (
            "🟠 Moderate – Your symptoms may still be early. Keep resting, hydrate, and avoid physical stress. "
            "Monitor carefully and consult a doctor if symptoms persist or worsen."
        )
    elif days <= 6:
        severity = (
            "🟡 Mild to Severe – It’s time to be cautious. Don’t ignore symptoms; seek medical attention soon. "
            "Even if symptoms are improving, it's best to rule out serious illness early."
        )
    else:
        severity = (
            "🔴 Severe – You’ve had these symptoms for more than a week. Please visit the nearest hospital immediately. "
            "Early medical attention saves lives. Don't delay your treatment. Your health matters most!"
        )

    return render_template("result.html", patient=patient_data, results=results, severity=severity)


@app.route("/restart")
def restart():
    patient_data.clear()
    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(debug=True)
