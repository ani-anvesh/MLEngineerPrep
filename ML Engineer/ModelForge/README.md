
# 🧠 Model Selection Pipeline (TF-IDF + GridSearch + MLflow)

This project implements a modular and configurable model selection pipeline in Python using:

- 🧹 Preprocessing
- 🧾 TF-IDF vectorization
- 🧠 Model training with `GridSearchCV`
- 📊 Tracking & logging with [MLflow](https://mlflow.org)
- ⚙️ Configuration via YAML

---

## 📁 Project Structure

```
.
├── config.yaml               # Model, vectorizer, and grid search settings
├── model_selection_pipeline.py
├── grid_results.json         # Full GridSearchCV results (logged by MLflow)
├── cv_results_summary.json   # Simplified results for quick inspection
└── README.md
```

---

## ✅ Setup

```bash
# Install dependencies
pip install -r requirements.txt
```

Sample `requirements.txt`:
```
scikit-learn
pandas
pyyaml
joblib
mlflow
```

---

## 🚀 How to Run

```bash
python model_selection_pipeline.py
```

This will:
- Load the config from `config.yaml`
- Run GridSearchCV using TF-IDF features
- Log all metrics, models, and artifacts to MLflow

---

## 📊 View MLflow UI

```bash
mlflow ui
```

Then go to: [http://localhost:5000](http://localhost:5000)

---

## ⚙️ YAML Configuration (`config.yaml`)

```yaml
vectorizer:
  max_features: 1000

model:
  type: random_forest
  hyperparameters:
    n_estimators: [100, 200]
    max_depth: [5, 10]

grid_search:
  cv: 5
  scoring: accuracy
```

Change the values to tune models without editing Python code.

---

## ✅ MLflow Logging

This pipeline logs:
- Best hyperparameters & score
- Trained model (`best_model.pkl`)
- Grid search results (`grid_results.json`)
- Summarized results (`cv_results_summary.json`)
- YAML config used for run

---

## 🛠️ Future Improvements

### 🌐 Push to Remote MLflow Server

1. Set the tracking URI:
   ```python
   mlflow.set_tracking_uri("http://<your-remote-mlflow-server>:5000")
   ```

2. Make sure the remote server has correct permissions and artifact storage (e.g., S3/GCS/MinIO).

---

### 🔁 Automatic Experiment Versioning

Add this to the script:
```python
experiment_name = "Model_Selection_Pipeline"
mlflow.set_experiment(experiment_name)

# Optional: print current version
experiment = mlflow.get_experiment_by_name(experiment_name)
print(f"Experiment: {experiment.name} (ID: {experiment.experiment_id})")
```

---

### 📦 Register the Model (MLflow Model Registry)

1. Add this to the MLflow run:
```python
mlflow.sklearn.log_model(
    sk_model=grid.best_estimator_,
    artifact_path="model",
    input_example=input_example,
    registered_model_name="TextClassificationModel"
)
```

2. This will create a versioned entry in MLflow Registry (ensure the server supports it).

---

## 🧠 License

MIT License — use, modify, and share freely.

---

## ✨ Contributions

Feel free to submit PRs for:
- New model types (e.g., SVM, Logistic Regression)
- Preprocessing improvements
- UI dashboard for config control
