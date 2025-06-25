import os
import shutil
import tempfile
import numpy as np
import pytest
import tensorflow as tf

from data_pipeline import createValidationData, validateCounts, preprocess, plot_training_history

def test_preprocess_normalizes_image():
    dummy_image = tf.random.uniform(shape=(128, 128, 3), minval=0, maxval=255, dtype=tf.int32)
    dummy_image = tf.cast(dummy_image, tf.uint8)
    dummy_label = tf.constant(1)

    image, label = preprocess(dummy_image, dummy_label)

    assert tf.reduce_max(image).numpy() <= 1.0
    assert tf.reduce_min(image).numpy() >= 0.0
    assert label.numpy() == 1

def test_plot_training_history(tmp_path):
    import matplotlib.pyplot as plt
    history = type('History', (object,), {})()
    history.history = {
        'loss': [0.8, 0.4],
        'val_loss': [0.9, 0.5],
        'accuracy': [0.5, 0.9],
        'val_accuracy': [0.4, 0.85],
    }

    # Redirect the savefig to a temporary path
    original_savefig = plt.savefig
    plt.savefig = lambda *args, **kwargs: None

    try:
        plot_training_history(history)
    finally:
        plt.savefig = original_savefig  # Restore

    assert True  # If no error, test passes

def test_create_validation_data_and_validate_counts():
    # Setup temp dirs
    with tempfile.TemporaryDirectory() as tmpdir:
        train_dir = os.path.join(tmpdir, "train")
        val_dir = os.path.join(tmpdir, "val")
        test_dir = os.path.join(tmpdir, "test")
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)

        # Create dummy class directories with images
        for cls in ['cat', 'dog']:
            os.makedirs(os.path.join(train_dir, cls))
            for i in range(10):
                with open(os.path.join(train_dir, cls, f'{cls}_{i}.jpg'), 'w') as f:
                    f.write("test")

        # Call function (this assumes it uses 'data/train' and 'data/val')
        current_dir = os.getcwd()
        try:
            os.chdir(tmpdir)
            os.makedirs("data", exist_ok=True)
            shutil.move("train", "data/train")
            shutil.move("test", "data/test")

            createValidationData()
            validateCounts()  # This should print counts, won't raise if correct
        finally:
            os.chdir(current_dir)

