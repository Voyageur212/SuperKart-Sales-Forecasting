"""SuperKart Flask prediction API | Author: Dimas Pradhana.

Run locally: python api.py
Health: GET /health
Single: POST /predict with JSON {"data": {...}}
Batch: POST /predict/batch with JSON {"data": [{...}, {...}]}
"""
import os
from pathlib import Path
import joblib
import pandas as pd
from flask import Flask, jsonify, request

app = Flask(__name__)
MODEL_PATH = Path(__file__).resolve().parent / "superkart_model.joblib"
model = joblib.load(MODEL_PATH)
FEATURES = list(model.feature_names_in_)
PERISHABLE = {"Meat", "Dairy", "Bread", "Breakfast", "Frozen Foods", "Fruits and Vegetables", "Seafood"}

def prepare(records):
    df = pd.DataFrame(records).copy()
    if "Product_Id_char" not in df and "Product_Id" in df:
        df["Product_Id_char"] = df["Product_Id"].astype(str).str[:2]
    if "Store_Age_Years" not in df and "Store_Establishment_Year" in df:
        df["Store_Age_Years"] = 2026 - pd.to_numeric(df["Store_Establishment_Year"], errors="coerce")
    if "Product_Type_Category" not in df and "Product_Type" in df:
        df["Product_Type_Category"] = df["Product_Type"].apply(
            lambda x: "Perishables" if x in PERISHABLE else "Non Perishables"
        )
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))
    data = df[FEATURES]
    if data.isna().any().any():
        raise ValueError("Input contains missing or invalid values")
    return data

@app.get("/health")
def health():
    return jsonify(status="ok", model="SuperKart tuned Random Forest", author="Dimas Pradhana")

@app.post("/predict")
def predict():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), dict):
        return jsonify(error='Expected JSON object with "data" as one record'), 400
    try:
        value = float(model.predict(prepare([payload["data"]]))[0])
        return jsonify(prediction=value)
    except (ValueError, TypeError, KeyError) as exc:
        return jsonify(error=str(exc)), 400

@app.post("/predict/batch")
def predict_batch():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), list) or not payload["data"] or not all(isinstance(x, dict) for x in payload["data"]):
        return jsonify(error='Expected JSON object with "data" as a nonempty list of records'), 400
    if len(payload["data"]) > 1000:
        return jsonify(error="Maximum 1000 records per request"), 400
    try:
        predictions = [float(v) for v in model.predict(prepare(payload["data"]))]
        return jsonify(predictions=predictions, count=len(predictions))
    except (ValueError, TypeError, KeyError) as exc:
        return jsonify(error=str(exc)), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
