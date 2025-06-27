# test_project.py

import os
import warnings
from tensorflow.keras.models import load_model
from src.data_pipeline import load_data
from src.evaluation import evaluate_model

warnings.filterwarnings("ignore")

def test_load_data():
    data_dir = "project/data"  # Adjust if needed

    if not os.path.exists(data_dir):
        print(f"❗ Skipping test_load_data: Directory {data_dir} does not exist.")
        return

    train, val = load_data(data_dir, img_size=(224, 224), batch_size=8)
    assert train.samples > 0, "No training samples loaded"
    assert val.samples > 0, "No validation samples loaded"
    assert len(train.class_indices) > 1, "Expected multiple classes"
    print("✅ test_load_data passed.")


def test_evaluate_model():
    model_path = "project/models/art_style_classifier.h5"
    data_dir = "project/data"
    output_dir = "project/outputs"

    if not os.path.exists(model_path):
        print("❗ Skipping test_evaluate_model: Model file missing.")
        return

    if not os.path.exists(data_dir):
        print("❗ Skipping test_evaluate_model: Data directory missing.")
        return

    _, val = load_data(data_dir, img_size=(224, 224), batch_size=8)
    class_names = list(val.class_indices.keys())

    model = load_model(model_path)
    metrics = evaluate_model(model, val, class_names, output_dir=output_dir)

    assert "accuracy" in metrics
    assert os.path.exists(os.path.join(output_dir, "classification_report.txt")), "Missing report"
    assert os.path.exists(os.path.join(output_dir, "confusion_matrix.png")), "Missing confusion matrix"
    print("✅ test_evaluate_model passed.")


if __name__ == "__main__":
    test_load_data()
    test_evaluate_model()
