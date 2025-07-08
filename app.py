# app.py
from flask import Flask, render_template, request, redirect, url_for, send_file
import pickle
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from fpdf import FPDF
import os

app = Flask(__name__)

# Load model and tools
model = load_model('diagnosis_model.h5')
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)
with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Load dataset
df = pd.read_csv('merged_symptoms.csv', encoding='latin1')
max_len = 30  # Set to same max_len as during training

# PDF Generator
class PrescriptionPDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 14)
        self.cell(0, 10, 'AI Health Diagnosis Prescription', ln=True, align='C')
        self.ln(10)

    def add_prescription(self, patient, diagnosis, precautions):
        self.set_font('Arial', '', 12)
        self.cell(0, 10, f"Name: {patient['name']}, Age: {patient['age']}, Gender: {patient['gender']}", ln=True)
        self.cell(0, 10, f"Location: {patient['location']}, Duration: {patient['duration']} days", ln=True)
        self.ln(5)
        for d in diagnosis:
            self.set_font('Arial', 'B', 12)
            self.cell(0, 10, f"Disease: {d['name']}", ln=True)
            self.set_font('Arial', '', 12)
            self.multi_cell(0, 10, f"Description: {d['description']}")
            self.cell(0, 10, "Precautions:", ln=True)
            for p in d['precautions']:
                self.cell(0, 10, f"- {p}", ln=True)
            self.ln(5)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        return redirect(url_for('diagnosis'))
    return render_template('index.html')

@app.route('/diagnosis', methods=['POST'])
def diagnosis():
    patient = {
        'name': request.form['name'],
        'age': request.form['age'],
        'gender': request.form['gender'],
        'duration': int(request.form['duration']),
        'location': request.form['location']
    }
    symptoms_input = request.form['symptoms'].lower()
    symptoms = symptoms_input.split(',')
    seq = tokenizer.texts_to_sequences([symptoms_input])
    padded = pad_sequences(seq, maxlen=max_len, padding='post')
    pred = model.predict(padded)[0]
    top5 = pred.argsort()[-5:][::-1]

    results = []
    for idx in top5:
        disease = label_encoder.inverse_transform([idx])[0]
        row = df[df['Disease'] == disease].iloc[0]
        precautions = [row[f'Precaution_{i}'] for i in range(1, 5) if pd.notna(row.get(f'Precaution_{i}', '')) and row.get(f'Precaution_{i}', '') != 'not specified']
        results.append({
            'name': disease,
            'description': row['Description'] if pd.notna(row['Description']) else 'Not available',
            'precautions': precautions,
            'confidence': f"{pred[idx]*100:.2f}%"
        })

    # Severity
    if patient['duration'] <= 3:
        severity = "Mild"
    elif 4 <= patient['duration'] <= 6:
        severity = "Moderate – You should consult a doctor."
    else:
        severity = "Severe – Urgent medical attention advised!"

    # Simulated hospital (can link to Google Maps API)
    hospital = f"General Hospital near {patient['location']}"

    # Save to session (or temporary file for PDF)
    request.session_data = (patient, results)

    return render_template('result.html', patient=patient, results=results, severity=severity, hospital=hospital)

@app.route('/download_pdf')
def download_pdf():
    patient, results = request.session_data
    pdf = PrescriptionPDF()
    pdf.add_page()
    pdf.add_prescription(patient, results, [])
    filepath = 'prescription.pdf'
    pdf.output(filepath)
    return send_file(filepath, as_attachment=True)

@app.route('/restart')
def restart():
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
