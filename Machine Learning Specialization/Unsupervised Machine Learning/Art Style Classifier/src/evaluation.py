from tensorflow.keras.models import Model
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import os

def extract_features(model, data_gen):
    feat_model = Model(inputs=model.input, outputs=model.layers[-2].output)
    features, labels, images = [], [], []

    for batch_x, batch_y in data_gen:
        f = feat_model.predict(batch_x)
        features.append(f)
        labels.append(batch_y)
        images.extend(batch_x)  # capture the raw images

        if len(features) * data_gen.batch_size >= data_gen.n:
            break

    return np.vstack(features), np.vstack(labels), np.array(images)


# 🎨 Recommend top-N visually similar artworks
def recommend_similar(features, index, top_n=5):
    """
    Given a matrix of feature vectors and an index, return the indices of
    the top-N most similar images (excluding itself).

    Args:
        features: numpy array of extracted feature vectors
        index: index of the query image
        top_n: number of similar images to return

    Returns:
        similar_indices: indices of top similar images
        similarity_scores: similarity values (cosine scores)
    """
    sims = cosine_similarity([features[index]], features)[0]
    similar_indices = sims.argsort()[-(top_n + 1):-1][::-1]  # Exclude self
    return similar_indices, sims[similar_indices]

def show_images(indices, image_array, title='Similar Images', path="outputs/similar_paintings.png"):
    fig, axes = plt.subplots(1, len(indices), figsize=(15, 5))
    for i, idx in enumerate(indices):
        img = image_array[idx]  # ✅ Correct way to access the image
        img = (img + 1) / 2     # Unnormalize if using MobileNetV2 preprocess
        axes[i].imshow(img)
        axes[i].axis('off')
    fig.suptitle(title)
    fig.savefig(path, bbox_inches='tight', dpi=300)
    plt.tight_layout()
    plt.show()

def evaluate_model(model, data_gen, class_names, output_dir="outputs", top_k_list=[2, 3]):
    os.makedirs(output_dir, exist_ok=True)

    y_true = data_gen.classes
    y_pred_proba = model.predict(data_gen, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)

    # Accuracy
    acc = accuracy_score(y_true, y_pred)
    print(f"✅ Accuracy: {acc:.4f}")

    # Classification report
    report = classification_report(y_true, y_pred, target_names=class_names)
    print("\n📊 Classification Report:")
    print(report)

    report_path = os.path.join(output_dir, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(f"Accuracy: {acc:.4f}\n\n")
        f.write(report)
    print(f"📁 Report saved to: {report_path}")

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title("🎨 Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_path)
    plt.show()
    print(f"🖼️ Confusion matrix saved to: {cm_path}")

    # Top-k accuracy
    def top_k_accuracy(k):
        return np.mean(np.any(np.argsort(y_pred_proba, axis=1)[:, -k:] == y_true.reshape(-1, 1), axis=1))

    metrics = {'accuracy': acc}
    for k in top_k_list:
        score = top_k_accuracy(k)
        metrics[f'top_{k}_accuracy'] = score
        print(f"🔁 Top-{k} Accuracy: {score:.4f}")

    return metrics