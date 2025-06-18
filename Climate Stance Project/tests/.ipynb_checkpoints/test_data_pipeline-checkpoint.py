import pytest
import pandas as pd
from sklearn.model_selection import train_test_split
from pipeline import clean_data, prepare_pipeline, train_model, evaluate_model, predict
from sklearn.ensemble import RandomForestClassifier

# Fixture: Dummy dataset
@pytest.fixture
def sample_data():
    data = {
        'tweet': [
            "Climate change is real and urgent.",
            "It's a hoax by the government.",
            "Not sure what to believe."
        ],
        'label': ['believer', 'denier', 'neutral']
    }
    return pd.DataFrame(data)

# Test preprocessing
def test_clean_data(sample_data):
    df = clean_data(sample_data.copy())
    assert not df.isnull().any().any(), "Preprocessing should remove NaNs"
    assert all(col in df.columns for col in ["tweet", "label"]), "Required columns missing"

# Test pipeline creation
def test_pipeline_build():
    clf = RandomForestClassifier()
    pipe = prepare_pipeline(clf)
    assert hasattr(pipe, "fit") and hasattr(pipe, "predict"), "Pipeline not correctly built"

# Test training
def test_training(sample_data):
    X = sample_data["tweet"]
    y = sample_data["label"]
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    pipe = prepare_pipeline(RandomForestClassifier())
    trained_pipe = train_model(pipe, X_train, y_train)
    assert trained_pipe is not None, "Training failed"

# Test prediction
def test_prediction(sample_data):
    X = sample_data["tweet"]
    y = sample_data["label"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipe = train_model(prepare_pipeline(RandomForestClassifier()), X_train, y_train)
    preds = predict(pipe, X_test)
    assert len(preds) == len(X_test), "Prediction size mismatch"

# Test evaluation (optional assert on F1 or Accuracy threshold)
def test_evaluation(sample_data):
    X = sample_data["tweet"]
    y = sample_data["label"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipe = train_model(prepare_pipeline(RandomForestClassifier()), X_train, y_train)
    metrics = evaluate_model(pipe, X_test, y_test)
    assert "accuracy" in metrics, "Evaluation must return accuracy"
