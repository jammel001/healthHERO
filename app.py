from flask import Flask, request, render_template
import pandas as pd
import joblib
import re

# Load model and data
clf = joblib.load("disease_model.pkl")
le = joblib.load("label_encoder.pkl")
merged_df = pd.read_csv("merged_data.csv")
symptom_cols = [col for col in merged_df.columns if col.lower().startswith("symptom_")]
all_symptoms = sorted(set(merged_df[symptom_cols].fillna('').values.ravel()) - {''})

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None

    if request.method == "POST":
        name = request.form['name']
        age = request.form['age']
        gender = request.form['gender']
        location = request.form['location']
        symptoms_input = request.form['symptoms']
        days = int(re.search(r'\d+', request.form['days']).group())

        symptoms = [s.strip().lower() for s in symptoms_input.split(",") if s.strip()]
        input_vec = [1 if s in symptoms else 0 for s in all_symptoms]
        pred = clf.predict([input_vec])[0]
        disease = le.inverse_transform([pred])[0]

        row = merged_df[merged_df['Disease'] == disease].iloc[0]
        description = row['Description']
        precautions = [row.get(f'Precaution_{i}') for i in range(1, 5) if pd.notna(row.get(f'Precaution_{i}'))]

        if days <= 3:
            severity = "🟢 Mild – Monitor and rest."
        elif days <= 6:
            severity = "🟡 Moderate – Consult a doctor."
        else:
            severity = "🔴 Severe – Visit hospital immediately."

        result = {
            "name": name,
            "disease": disease,
            "description": description,
            "precautions": precautions,
            "severity": severity
        }

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
