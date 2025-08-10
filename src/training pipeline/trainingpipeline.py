import pandas as pd
import numpy as np
import pickle
import json
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import mlflow
import optuna
import os
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.path.join(BASE_DIR, "model")

# 1. Load the dataset from CSV
def load_data(path):
    return pd.read_csv(path)

# 2. Split the dataset into training and testing sets
def split_data(df, target_column):
    X = df.drop(columns=[target_column])  # Features
    y = df[target_column]  # Target variable
    return train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Train multiple baseline models and store their scores
def train_models(X_train, y_train, X_test, y_test, dataset_version="v1.0"):
    models = {
        "SVR": SVR(),
        "DecisionTree": DecisionTreeRegressor(),
        "RandomForest": RandomForestRegressor(),
        "XGBoost": XGBRegressor()
    }
    scores = {}

    for name, model in models.items():
        start_time = time.time()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        score = r2_score(y_test, preds)
        scores[name] = (score, model)

        # Log baseline model in MLflow
        mlflow.set_experiment("Student Productivity Prediction")
        with mlflow.start_run(run_name=f"Baseline_{name}"):
            mlflow.log_param("model_name", name)
            mlflow.log_param("dataset_version", dataset_version)
            mlflow.log_metric("r2_score", score)
            mlflow.log_metric("training_time_sec", time.time() - start_time)
            mlflow.sklearn.log_model(model, artifact_path=name)
            mlflow.set_tags({"stage": "baseline", "developer": "Gaythri"})

    return scores

# 4. Get the best performing model based on R² score
def get_best_model(scores):
    return max(scores.items(), key=lambda x: x[1][0])

# 5. Hyperparameter tuning for all models using Optuna
def tune_model(model_name, X_train, y_train):
    # Define the Optuna objective function
    def objective(trial):
        if model_name == "SVR":
            params = {
                "C": trial.suggest_float("C", 0.1, 10.0),
                "kernel": trial.suggest_categorical("kernel", ["linear", "rbf", "poly"]),
                "gamma": trial.suggest_categorical("gamma", ["scale", "auto"])
            }
            model = SVR(**params)

        elif model_name == "DecisionTree":
            params = {
                "max_depth": trial.suggest_int("max_depth", 2, 20),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 10)
            }
            model = DecisionTreeRegressor(**params)

        elif model_name == "RandomForest":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "max_depth": trial.suggest_int("max_depth", 2, 20),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 10)
            }
            model = RandomForestRegressor(**params)

        elif model_name == "XGBoost":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "max_depth": trial.suggest_int("max_depth", 2, 20),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3)
            }
            model = XGBRegressor(**params)

        # Perform cross-validation to evaluate performance
        score = cross_val_score(model, X_train, y_train, cv=3, scoring="r2").mean()
        return score

    # Run Optuna study
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)  # 20 tuning iterations
    best_params = study.best_params  # Best found hyperparameters

    # Train final model with best params
    if model_name == "SVR":
        best_model = SVR(**best_params)
    elif model_name == "DecisionTree":
        best_model = DecisionTreeRegressor(**best_params)
    elif model_name == "RandomForest":
        best_model = RandomForestRegressor(**best_params)
    elif model_name == "XGBoost":
        best_model = XGBRegressor(**best_params)

    best_model.fit(X_train, y_train)  # Retrain on full training data
    return best_model, best_params

# 6. Save model as .pkl
def save_model(model):
    with open(os.path.join(MODEL_DIR, "best_model.pkl"), "w") as f:
        pickle.dump(model, f)

# 7. Save expected column names for inference pipeline
def save_expected_columns(columns):
    with open(os.path.join(MODEL_DIR, "expected_columns.json"), "w") as f:
        json.dump(columns, f)

# 8. Log metrics and parameters to MLflow
def log_tuned_model(model, model_name, score, params, dataset_version="v1.0"):
    mlflow.set_experiment("Student Productivity Prediction")
    with mlflow.start_run(run_name=f"Tuned_{model_name}"):
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("dataset_version", dataset_version)
        mlflow.log_params(params)
        mlflow.log_metric("r2_score", score)
        mlflow.sklearn.log_model(model, artifact_path=model_name)
        mlflow.set_tags({"stage": "tuned_model", "developer": "Gaythri"})

# 9. Full pipeline
def run_pipeline(data_path, target_column, tune_only=False):
    df = load_data(data_path)
    df = df.drop(columns=["Student_ID", "Student_Name"])
    df = pd.get_dummies(df, drop_first=True)

    X_train, X_test, y_train, y_test = split_data(df, target_column)
    save_expected_columns(X_train.columns.tolist())

    if not tune_only:
        scores = train_models(X_train, y_train, X_test, y_test)
        best_model_name, (best_score, best_model) = get_best_model(scores)
        print(f"Best model before tuning: {best_model_name} - R²: {best_score:.4f}")
    else:
        with open("model/last_best_model.txt", "r") as f:
            best_model_name = f.read().strip()

    tuned_model, best_params = tune_model(best_model_name, X_train, y_train)
    tuned_preds = tuned_model.predict(X_test)
    tuned_score = r2_score(y_test, tuned_preds)

    log_tuned_model(tuned_model, best_model_name, tuned_score, best_params)
    save_model(tuned_model)

    with open("model/last_best_model.txt", "w") as f:
        f.write(best_model_name)

    print(f"Tuned {best_model_name} R²: {tuned_score:.4f}")

# 10. Run the full pipeline
if __name__ == "__main__":
    run_pipeline("data/processed_data.csv", "Productivity_Score")