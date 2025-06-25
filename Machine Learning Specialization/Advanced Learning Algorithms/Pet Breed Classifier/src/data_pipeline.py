import os, shutil
import random
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
output_dir = os.path.join('../', 'outputs')


def createValidationData():
    # Config
    train_path = 'data/train'   # <-- corrected path
    val_path = 'data/val'       # <-- corrected path
    split_ratio = 0.2  # 20% of train goes to val
    
    # Ensure val directory exists
    for label in ['cats', 'dogs']:
        os.makedirs(os.path.join(val_path, label), exist_ok=True)
    
    # Move 20% from train to val
    for label in ['cats', 'dogs']:
        source_dir = os.path.join(train_path, label)
        dest_dir = os.path.join(val_path, label)
    
        images = os.listdir(source_dir)
        random.shuffle(images)
        split_size = int(len(images) * split_ratio)
        val_images = images[:split_size]
    
        for img in val_images:
            shutil.move(os.path.join(source_dir, img), os.path.join(dest_dir, img))
    
    print("✅ Validation set created successfully.")

def validateCounts():
    base_dir = 'data'
    for split in ['train', 'val', 'test']:
        print(f"\n{split} data:")
        for category in ['cats', 'dogs']:
            path = os.path.join(base_dir, split, category)
            count = len(os.listdir(path))
            print(f"  {category}: {count} images")
    

def load_data(path):
    """Loads dataset from the given path."""


def preprocess(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

def plot_training_history(history):
    plt.figure(figsize=(12,5))

    # Loss plot
    plt.subplot(1,2,1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.legend()
    plt.title('Loss over epochs')

    # Accuracy plot
    plt.subplot(1,2,2)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.legend()
    plt.title('Accuracy over epochs')
    plt.savefig(os.path.join('outputs', 'Loss and Accuracy over epochs.png'))

    plt.show()


