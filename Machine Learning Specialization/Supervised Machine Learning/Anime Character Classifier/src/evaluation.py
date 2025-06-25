# src/evaluation.py

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import classification_report
import os



def evaluate_classification(y_true, y_pred, average='macro'):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average=average, zero_division=0),
        "recall": recall_score(y_true, y_pred, average=average, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, average=average, zero_division=0),
    }

def evaluate_roc_auc(y_true, y_scores, multi_class='ovr'):
    """Note: y_scores = probability scores, not labels"""
    return roc_auc_score(y_true, y_scores, multi_class=multi_class, average='macro')

def save_classification_reports(y_test, y_pred,output_path):
    report = classification_report(y_test, y_pred)
    file_path = os.path.join(output_path, 'classification_report.txt')
    # Write to the file
    with open(file_path, 'w') as f:
        f.write(report)
        
    print(report)