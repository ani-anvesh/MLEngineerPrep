from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report
)
import os
import numpy as np


def evaluate_classification(y_true, y_pred, average='macro'):
    """
    Compute accuracy, precision, recall, and F1-score.

    Parameters:
        y_true (array-like): True class labels.
        y_pred (array-like): Predicted class labels.
        average (str): Averaging method for multi-class metrics.

    Returns:
        dict: Dictionary with accuracy, precision, recall, and F1.
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average=average, zero_division=0),
        "recall": recall_score(y_true, y_pred, average=average, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, average=average, zero_division=0),
    }


def evaluate_roc_auc(y_true, y_proba, multi_class='ovr', average='macro'):
    """
    Compute ROC-AUC for multi-class classification.

    Parameters:
        y_true (array-like): True class labels.
        y_proba (array-like): Probability scores (n_samples, n_classes).
        multi_class (str): 'ovr' or 'ovo'.
        average (str): Averaging method.

    Returns:
        float: ROC-AUC score.
    """
    try:
        return roc_auc_score(y_true, y_proba, multi_class=multi_class, average=average)
    except ValueError as e:
        print(f"[Warning] ROC-AUC could not be computed: {e}")
        return np.nan


def save_classification_reports(y_true, y_pred, labels=None, output_dir="outputs"):
    """
    Save classification report to a file and print to console.

    Parameters:
        y_true (array-like): True class labels.
        y_pred (array-like): Predicted class labels.
        labels (list, optional): Class label names.
        output_dir (str): Directory to save report.
    """
    report = classification_report(y_true, y_pred, target_names=labels)
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, 'classification_report.txt')

    with open(file_path, 'w') as f:
        f.write(report)

    print(report)
