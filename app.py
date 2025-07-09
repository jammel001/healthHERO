from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np
from keras.layers import TFSMLayer
from keras import Input, Model
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

# Load model components
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

df = pd.read_csv("merged_symptoms.csv", encoding="latin1")

defined_symptoms = df[['Symptoms.1', 'General Explanation', 'Medical Explanation']].dropna()
explained_symptoms = {
    row['Symptoms.1'].strip().lower(): {
        "general": row['General Explanation'],
        "medical": row['Medical Explanation']
    }
    for _, row in defined_symptoms.iterrows()
}

max_len = max(len(seq) for seq in tokenizer.texts_to_sequences(df['Symptoms'].astype(str)))

# Load model
layer = TFSMLayer("diagnosis_model", call_endpoint="serving_default")
inputs = Input(shape=(max_len,), dtype="float32")
outputs = layer(inputs)
model = Model(inputs, outputs)

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
            explained.append((s, "Not found", "Not found"))

    confirm = request.form.get('confirm')
    if confirm != "yes":
        return render_template("result.html", name=name, canceled=True)

    text = ", ".join(confirmed_symptoms)
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=max_len, padding='post').astype("float32")
    raw_output = model.predict(padded)
    pred = list(raw_output.values())[0][0]
    top5 = pred.argsort()[-5:][::-1]

    predictions = []
    for idx in top5:
        disease = label_encoder.inverse_transform([idx])[0]
        row = df[df['Disease'] == disease].iloc[0]
        desc = row['Description'] if pd.notna(row['Description']) else "No description"
        precautions = [row.get(f"Precaution_{i}", "") for i in range(1, 5) if pd.notna(row.get(f"Precaution_{i}"))]
        predictions.append({
            "disease": disease,
            "confidence": f"{pred[idx]*100:.2f}%",
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
