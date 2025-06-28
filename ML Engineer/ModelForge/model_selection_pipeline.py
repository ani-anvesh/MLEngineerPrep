import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import yaml
import mlflow
import mlflow.sklearn
import joblib
import json

# --- Load Config ---
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# --- Sample Data (replace with actual loading) ---
# Dummy data for testing
data = {
    "text": [
        "I love machine learning",
        "Random forests are great",
        "Text data is fun",
        "ML models need tuning",
        "Natural language processing is powerful",
        "Sklearn is easy to use",
        "Hyperparameter tuning is critical",
        "MLflow tracks experiments",
        "Transformers are state of the art",
        "Data preprocessing is essential"
    ],
    "label": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
}

df = pd.DataFrame(data)

# --- Preprocessing (optional for now) ---
def preprocess(text_series):
    return text_series.str.lower()

df["text"] = preprocess(df["text"])

# --- Vectorizer ---
vectorizer = TfidfVectorizer(max_features=config["vectorizer"]["max_features"])

# --- Model Selection ---
def get_model(model_type):
    if model_type == "random_forest":
        return RandomForestClassifier()
    raise ValueError(f"Unsupported model type: {model_type}")

model = get_model(config["model"]["type"])

# --- Build Pipeline ---
pipeline = Pipeline([
    ("vectorizer", vectorizer),
    ("clf", model)
])

# Format param grid with correct prefix for Pipeline
param_grid = {f"clf__{k}": v for k, v in config["model"]["hyperparameters"].items()}

# --- Grid Search ---
grid = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=config["grid_search"]["cv"],
    scoring=config["grid_search"]["scoring"],
    verbose=1,
    n_jobs=-1
)

X = df["text"]
y = df["label"]

# Start MLflow logging
mlflow.set_experiment("Model_Selection_Pipeline")

with mlflow.start_run():
    grid.fit(X, y)

    print("Logging model and results to MLflow...")

    # Log best parameters
    mlflow.log_params(grid.best_params_)
    mlflow.log_metric("best_score", grid.best_score_)

    # Log model with input example
    input_example = pd.DataFrame({"text": ["This is an example input"]})
    mlflow.sklearn.log_model(grid.best_estimator_, artifact_path="model", input_example=input_example)

    # Save and log full cv_results_
    serializable_results = {
        k: v.tolist() if isinstance(v, np.ndarray) else v
        for k, v in grid.cv_results_.items()
    }
    with open("grid_results.json", "w") as f:
        json.dump(serializable_results, f, indent=2)
    mlflow.log_artifact("grid_results.json")

    # Summarized results
    results_df = pd.DataFrame(grid.cv_results_)
    summary_df = results_df[["params", "mean_test_score", "std_test_score", "rank_test_score"]]
    mlflow.log_dict(summary_df.to_dict(orient="records"), "cv_results_summary.json")

    # Log config
    mlflow.log_artifact("config.yaml")

