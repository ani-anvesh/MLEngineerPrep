from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report
)
import os

def evaluate_classification(y_true, y_pred, average='macro'):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average=average, zero_division=0),
        "recall": recall_score(y_true, y_pred, average=average, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, average=average, zero_division=0),
    }


def evaluate_roc_auc(y_true, y_scores, multi_class='ovr'):
    """Note: y_scores = probability scores, not hard labels"""
    return roc_auc_score(y_true, y_scores, multi_class=multi_class, average='macro')


def save_classification_reports(y_true, y_pred):
    report = classification_report(y_true, y_pred, target_names=["Cat", "Dog"])
    os.makedirs('outputs', exist_ok=True)
    file_path = os.path.join('outputs', 'classification_report.txt')
    with open(file_path, 'w') as f:
        f.write(report)
    print(report)
