import pandas as pd
import zipfile
from io import TextIOWrapper
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity


def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

def read_jsons_from_zip(zip_path):
    dataframes = []
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for file_name in zip_ref.namelist():
            if file_name.endswith('.json'):
                with zip_ref.open(file_name) as file:
                    # Use TextIOWrapper to handle bytes stream
                    df = pd.read_json(TextIOWrapper(file, encoding='utf-8'))
                    dataframes.append(df)
    
    return pd.concat(dataframes, ignore_index=True)

def preprocess_ingredients(df):
    df['joined_ingredients'] = df['ingredients'].apply(lambda x: " ".join(x))
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df['joined_ingredients'])
    tfidf_array = tfidf_matrix.toarray()
    return df, tfidf_matrix, tfidf_array, vectorizer

def kmeans_scratch(X, k, max_iters=100, tol=1e-4, random_state=42):
    np.random.seed(random_state)

    # Step 1: Initialize centroids randomly from data points
    initial_idx = np.random.choice(len(X), k, replace=False)
    centroids = X[initial_idx]

    for _ in range(max_iters):
        # Step 2: Assign clusters
        labels = np.array([np.argmin([euclidean_distance(x, c) for c in centroids]) for x in X])

        # Step 3: Compute new centroids
        new_centroids = np.array([X[labels == j].mean(axis=0) for j in range(k)])

        # Step 4: Check for convergence (centroid movement < tolerance)
        if np.all(np.linalg.norm(new_centroids - centroids, axis=1) < tol):
            break

        centroids = new_centroids

    return centroids, labels

def cluster_recipes_scratch(array, k):
    centroids, labels = kmeans_scratch(array, k)
    return centroids, labels

def cluster_recipes(matrix, k):
    model = KMeans(n_clusters=k, random_state=42)
    labels = model.fit_predict(matrix)
    return model, labels

def compute_inertia(X, centroids, labels):
    # Sum of squared distances of points to their assigned centroid
    inertia = 0.0
    for i, centroid in enumerate(centroids):
        points = X[labels == i]
        if len(points) > 0:
            distances = np.linalg.norm(points - centroid, axis=1) ** 2
            inertia += distances.sum()
    return inertia

def elbow_method_scratch(matrix, max_k=10):
    inertia = []
    for k in range(1, max_k + 1):
        centroids, labels = kmeans_scratch(matrix, k)
        inertia_k = compute_inertia(matrix, centroids, labels)
        inertia.append(inertia_k)
    return inertia, centroids, labels

def elbow_method(tfidf_matrix):
    inertias = []
    for k in range(2, 15):
        km = KMeans(n_clusters=k).fit(tfidf_matrix)
        inertias.append(km.inertia_)
    return inertias

def reduce_pca(tfidf_array):
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(tfidf_array)
    return reduced

def get_similar_recipes(index, matrix, top_n=5):
    cosine_sim = cosine_similarity(matrix[index], matrix)
    sim_scores = list(enumerate(cosine_sim[0]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    return sim_scores[1:top_n+1]
