from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import re

# Initialize Flask app
app = Flask(__name__)

# Load model and resources
clf = joblib.load("disease_model.pkl")
le = joblib.load("label_encoder.pkl")
merged_df = pd.read_csv("merged_data.csv")

# Prepare list of all possible symptoms
symptom_cols = [col for col in merged_df.columns if col.lower().startswith("symptom_")]
all_symptoms = sorted(set(merged_df[symptom_cols].fillna('').values.ravel()) - {''})

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/diagnose", methods=["POST"])
def diagnose():
    data = request.get_json()
    name = data.get("name")
    age = data.get("age")
    gender = data.get("gender")
    location = data.get("location")
    symptoms_input = data.get("symptoms", "")
    days = int(re.search(r'\d+', data.get("days", "0")).group())

    # Preprocess input symptoms
    symptoms = [s.strip().lower() for s in symptoms_input.split(",") if s.strip()]
    input_vec = [1 if s in symptoms else 0 for s in all_symptoms]

    # Predict disease
    prediction = clf.predict([input_vec])[0]
    disease = le.inverse_transform([prediction])[0]

    # Retrieve disease data
    row = merged_df[merged_df["Disease"] == disease].iloc[0]
    description = row["Description"]
    precautions = [row.get(f"Precaution_{i}") for i in range(1, 5) if pd.notna(row.get(f"Precaution_{i}"))]

    # Severity level messages
    if days <= 3:
        severity = (
            "🟢 *Mild symptoms detected.*\n"
            "Your condition appears mild at this stage. It’s important to rest, stay hydrated, and monitor your symptoms carefully. "
            "Early care can make a big difference in recovery. Keep an eye on any changes, and take care of yourself."
        )
    elif days <= 6:
        severity = (
            "🟡 *Moderate symptoms detected.*\n"
            "You’ve been feeling unwell for several days now. It is advisable to consult a doctor soon to ensure your condition doesn’t worsen. "
            "Medical guidance can help you recover faster and avoid complications. Take it seriously and act in time."
        )
    else:
        severity = (
            "🔴 *Severe symptoms detected.*\n"
            "Your symptoms have persisted for more than a week. This is a strong indicator of a potential health risk. "
            "It is critical to seek immediate medical attention. Please do not delay — your well-being matters greatly. Visit the nearest hospital now."
        )

    # Health tip
    health_tip = (
        "🌿 *Health Tip:*\n"
        "Maintain a balanced diet, drink enough water, sleep at least 7 hours daily, and reduce stress. "
        "These habits help strengthen your immune system and overall health."
    )

    # Add hospital visit recommendation
    location_message = f"🏥 You are strongly advised to visit a nearby hospital in **{location}** for professional consultation and diagnosis."

    # Final response
    return jsonify({
        "name": name,
        "age": age,
        "gender": gender,
        "location": location,
        "disease": disease,
        "description": description,
        "precautions": precautions,
        "severity": severity,
        "location_advice": location_message,
        "health_tip": health_tip,
        "followup": "Would you like to make another diagnosis?"
    })

if __name__ == "__main__":
    app.run(debug=True)
