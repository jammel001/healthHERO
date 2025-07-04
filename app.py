from flask import Flask, request, jsonify
import pandas as pd
import joblib

# Load trained model and data
clf = joblib.load("disease_model.pkl")
le = joblib.load("label_encoder.pkl")
merged_df = pd.read_csv("merged_data.csv")

# Extract symptom columns and all unique symptoms
symptom_cols = [col for col in merged_df.columns if col.lower().startswith("symptom_")]
all_symptoms = sorted(set(merged_df[symptom_cols].fillna('').values.ravel()) - {''})

app = Flask(__name__)

@app.route("/")
def home():
    return "🩺 AI Health Diagnosis API by Bara'u Magaji Kanya"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    # Required fields
    name = data.get("name", "Patient")
    age = data.get("age", "")
    gender = data.get("gender", "")
    location = data.get("location", "")
    symptoms = data.get("symptoms", [])
    days = int(data.get("days", 0))

    # Encode symptoms
    input_vec = [1 if symptom.lower() in symptoms else 0 for symptom in all_symptoms]
    prediction = clf.predict([input_vec])[0]
    disease = le.inverse_transform([prediction])[0]

    # Match most likely row
    row = merged_df[merged_df['Disease'] == disease].iloc[0]
    description = row['Description']
    precautions = [row.get(f'Precaution_{i}') for i in range(1, 5) if pd.notna(row.get(f'Precaution_{i}'))]

    # Severity
    if days <= 3:
        severity = "Mild – Monitor your health."
    elif days <= 6:
        severity = "Moderate – Please consult a doctor."
    else:
        severity = "Severe – Visit a hospital immediately."

    return jsonify({
        "name": name,
        "predicted_disease": disease,
        "description": description,
        "precautions": precautions,
        "severity": severity,
        "location": location,
        "disclaimer": "This AI does not replace professional medical advice."
    })

if __name__ == "__main__":
    app.run(debug=True)
