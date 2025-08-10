from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import pickle
import json
import os

# Load model and expected columns
base_dir = os.path.dirname(__file__)
MODEL_PATH = os.path.abspath(os.path.join(base_dir, "..", "..", "model", "best_model.pkl"))
COLUMNS_PATH = os.path.abspath(os.path.join(base_dir, "..", "..", "model", "expected_columns.json"))

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(COLUMNS_PATH, "r") as f:
    expected_columns = json.load(f)

# FastAPI app
app = FastAPI(title="Student Productivity Prediction API")

# Request schema
class StudentData(BaseModel):
    Grade: str
    Section: str
    Maths: float
    Computer_Science: float
    English: float
    Tamil: float
    Physics: float
    Chemistry: float
    WhatsApp_Time: float
    Insta_Reel_Freq: float
    YouTube_Entertainment: float
    Study_Time: float
    Call_Chat_Time: float

@app.post("/predict")
def predict(data: StudentData):
    # Convert input to DataFrame
    input_df = pd.DataFrame([data.dict()])
    input_df = pd.get_dummies(input_df, drop_first=True)

    # Align columns with training
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[expected_columns]

    # Prediction
    prediction = model.predict(input_df)[0]
    prediction = float(prediction)
    if prediction < 0:
        prediction = 0.0  # Avoid negative productivity scores

    return {"Predicted Productivity Score": prediction}