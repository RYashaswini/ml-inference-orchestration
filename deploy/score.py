import os
import joblib
import json
import pandas as pd

model = None

def init():
    global model
    model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR"), "attrition_model.joblib")
    model = joblib.load(model_path)

def run(raw_data):
    data = json.loads(raw_data)
    df = pd.DataFrame([data])
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]
    return json.dumps({
        "prediction": int(prediction),
        "probability": round(float(probability), 3)
    })