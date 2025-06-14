import numpy as np
import pandas as pd
from src.evaluation import evaluate_classification
from src import data_pipeline

def test_load_data():
    df = data_pipeline.load_data('tests/sample_data.csv')
    assert isinstance(df, pd.DataFrame)
    assert not df.empty

def test_evaluate_all_zeros():
    y_true = [0, 0, 0]
    y_pred = [0, 0, 0]
    metrics = evaluate_classification(y_true, y_pred)
    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1_score"] == 1.0

def test_evaluate_perfect_multiclass():
    y_true = [0, 1, 2, 0, 1, 2]
    y_pred = [0, 1, 2, 0, 1, 2]
    metrics = evaluate_classification(y_true, y_pred)
    assert metrics["accuracy"] == 1.0
    assert metrics["f1_score"] == 1.0

def test_evaluate_partial_mismatch():
    y_true = [0, 1, 2, 2]
    y_pred = [0, 2, 2, 1]
    metrics = evaluate_classification(y_true, y_pred)
    assert 0 <= metrics["accuracy"] <= 1
    assert 0 <= metrics["f1_score"] <= 1

def test_evaluate_class_imbalance():
    y_true = [0] * 100 + [1] * 5 + [2] * 5
    y_pred = [0] * 100 + [0] * 10  # completely misclassified minority
    metrics = evaluate_classification(y_true, y_pred)
    assert metrics["accuracy"] > 0.7  # still high due to imbalance
    assert metrics["recall"] < 0.5    # minority classes ignored
    assert metrics["precision"] < 0.5

def test_evaluate_with_empty_prediction():
    y_true = [0, 1, 2]
    y_pred = [None, None, None]
    y_pred_clean = [0 if p is None else p for p in y_pred]
    metrics = evaluate_classification(y_true, y_pred_clean)
    assert isinstance(metrics, dict)

def test_invalid_labels_graceful():
    y_true = [0, 1, 2]
    y_pred = ['a', 'b', 'c']  # invalid labels
    try:
        evaluate_classification(y_true, y_pred)
        assert False, "Expected exception due to invalid labels"
    except ValueError:
        assert True
