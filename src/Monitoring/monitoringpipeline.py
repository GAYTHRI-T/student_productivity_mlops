import pandas as pd
import pickle
import json
from sklearn.metrics import r2_score
import subprocess
import os

base_dir = os.path.dirname(__file__)
MODEL_PATH = os.path.abspath(os.path.join(base_dir, "..", "..", "model", "best_model.pkl"))
COLUMNS_PATH = os.path.abspath(os.path.join(base_dir, "..", "..", "model", "expected_columns.json"))

# Thresholds
TUNE_THRESHOLD = 0.80
RETRAIN_THRESHOLD = 0.70

def monitor_model(new_data_path, target_column):
    # Load latest model
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    with open(COLUMNS_PATH, "r") as f:
        expected_columns = json.load(f)

    # Load new dataset
    df = pd.read_csv(new_data_path)
    df = df.drop(columns=["Student_ID", "Student_Name"])
    df = pd.get_dummies(df, drop_first=True)

    # Align columns
    for col in expected_columns:
        if col not in df.columns:
            df[col] = 0
    df = df[expected_columns + [target_column]]

    X = df.drop(columns=[target_column])
    y = df[target_column]

    preds = model.predict(X)
    score = r2_score(y, preds)
    print(f"Current R²: {score:.4f}")

    if score < RETRAIN_THRESHOLD:
        print("Model performance low — retraining...")
        subprocess.run(["python", "trainingpipeline.py"])
    elif score < TUNE_THRESHOLD:
        print("Model performance degraded — tuning parameters...")
        subprocess.run(["python", "trainingpipeline.py", "--tune_only"])
    else:
        print("Model performance acceptable. No action taken.")

if __name__ == "__main__":
    monitor_model("data/new_data.csv", "Productivity_Score")