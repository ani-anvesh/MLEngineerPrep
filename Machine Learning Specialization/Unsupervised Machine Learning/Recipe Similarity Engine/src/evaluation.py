import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import davies_bouldin_score


def plot_elbow(inertias):
    plt.figure(figsize=(8, 4))
    plt.plot(range(2, 2 + len(inertias)), inertias, marker='o')
    plt.title("Elbow Method for Optimal k")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Inertia")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("outputs/elbow_plot.png", dpi=300)
    plt.show()


def plot_pca(reduced, labels):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        x=reduced[:, 0],
        y=reduced[:, 1],
        hue=labels,
        palette='Set2',
        legend='full'
    )
    plt.title('PCA of Recipes Colored by Cluster')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.tight_layout()
    plt.savefig("outputs/pca_clusters.png", dpi=300)
    plt.show()


def classification_report(tfidf_array, labels):
    sil = silhouette_score(tfidf_array, labels)
    ch = calinski_harabasz_score(tfidf_array, labels)
    db = davies_bouldin_score(tfidf_array, labels)

    report = (
        f"Silhouette Score: {sil:.4f}\n"
        f"Calinski-Harabasz Index: {ch:.4f}\n"
        f"Davies-Bouldin Index: {db:.4f}\n"
    )

    print(report)

    with open("outputs/cluster_metrics.txt", "w") as f:
        f.write(report)