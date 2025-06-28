# tests/test_evaluation.py

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

def evaluate_model(model, x_val, y_val_labels):
    y_pred_probs = model.predict(x_val)
    y_preds = np.argmax(y_pred_probs, axis=1)

    report = classification_report(
        y_val_labels,
        y_preds,
        labels=[0, 1, 2],
        target_names=["High", "Medium", "Low"]
    )
    print("Classification Report:\n", report)

    cm = confusion_matrix(y_val_labels, y_preds, labels=[0,1,2])
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=["High", "Medium", "Low"],
                yticklabels=["High", "Medium", "Low"])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()

if __name__ == "__main__":
    # Dummy test data: 5 samples, shape must match your model input shape (12, 207, 2)
    x_val = np.random.rand(5, 12, 207, 2).astype(np.float32)

    # Dummy true labels for 5 samples, with all classes represented
    y_val_labels = np.array([0, 1, 2, 1, 0])

    # Dummy model that randomly predicts 3 classes
    # Replace this with loading your real model like:
    # model = tf.keras.models.load_model('models/your_model.h5')
    class DummyModel:
        def predict(self, x):
            # Return uniform random probabilities over 3 classes for each sample
            batch_size = x.shape[0]
            return np.random.rand(batch_size, 3)
    
    model = DummyModel()

    evaluate_model(model, x_val, y_val_labels)
