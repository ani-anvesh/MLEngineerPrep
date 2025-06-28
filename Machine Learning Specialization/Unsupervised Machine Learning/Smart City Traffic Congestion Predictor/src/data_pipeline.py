import numpy as np

def load_metr_la_data(split="train"):
    path = f"data/METR-LA/{split}.npz"
    with np.load(path) as data:
        x, y = data['x'], data['y']
    return x, y

def convert_speed_to_congestion_label(y, low_thresh=30, high_thresh=50):
    """
    y: (samples, time, sensors, features) → speed values
    Returns: (samples,) congestion class per sample: 0=high, 1=medium, 2=low
    """
    # You want to average over time (axis=1), sensors (axis=2), and features (axis=3)
    avg_speed = y.mean(axis=(1, 2, 3))

    # Convert to class labels
    labels = np.digitize(avg_speed, bins=[low_thresh, high_thresh])  # 0, 1, 2
    return labels

