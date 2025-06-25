# 🍽️ Recipe Similarity Engine

This project is a **content-based recipe similarity engine** that clusters recipes using their ingredients and returns similar recipes based on cosine similarity. It includes visualization, evaluation metrics, and a custom implementation of the KMeans algorithm.

---

## 📌 Features

- Preprocess and vectorize recipe ingredients using TF-IDF
- Cluster recipes using custom or sklearn KMeans
- Visualize clusters with PCA and elbow method plots
- Evaluate clustering using:
  - Silhouette Score
  - Calinski-Harabasz Index
  - Davies-Bouldin Index
- Find top-N similar recipes using cosine similarity
- Outputs saved to `outputs/` folder for reproducibility

---

## 🗂️ Project Structure

```
project/
├── data/                 # Input datasets
├── models/               # Saved models (KMeans, TF-IDF vectorizer)
├── notebooks/            # Jupyter notebooks for exploration
├── outputs/              # Generated plots & evaluation reports
├── src/                  # Source code (pipeline, clustering, evaluation)
├── tests/                # Unit tests
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

---

## 🧪 Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
```

---

## 🧼 Preprocessing

Recipes are preprocessed using the `joined_ingredients` column. TF-IDF vectorization is applied on this field.

---

## 🤖 Clustering & Evaluation

Use either **sklearn** or a **custom KMeans implementation**.

### Elbow Method

```python
from src.evaluation import elbow_method_scratch, plot_elbow

inertias = elbow_method_scratch(tfidf_array, max_k=10)
plot_elbow(inertias)  # Saved to outputs/elbow_plot.png
```

### PCA Plot

```python
from src.evaluation import plot_pca

plot_pca(pca_reduced, labels)  # Saved to outputs/pca_clusters.png
```

### Clustering Metrics

```python
from src.evaluation import classification_report

classification_report(tfidf_array, labels)
# Output written to outputs/cluster_metrics.txt
```

---

## 🔍 Recipe Similarity

Use cosine similarity to find similar recipes:

```python
from src.similarity import get_similar_recipes

index = df[df['id'] == 10550].index[0]
similar_recipes = get_similar_recipes(index, tfidf_array, top_n=5)
```

---

## 📊 Outputs

| File                        | Description                            |
|----------------------------|----------------------------------------|
| `outputs/elbow_plot.png`   | Elbow curve for determining best k     |
| `outputs/pca_clusters.png` | PCA 2D scatter plot of recipe clusters |
| `outputs/cluster_metrics.txt` | Evaluation metrics for clustering |

---

## 🧪 Tests

Run test cases:

```bash
pytest tests/
```

---

## 📚 Dataset

Use the [Kaggle Recipe Ingredients Dataset](https://www.kaggle.com/kaggle/recipe-ingredients-dataset) or your custom cleaned dataset.

Make sure it contains a `joined_ingredients` column (stringified ingredient lists) before TF-IDF vectorization.

---

## 📬 License

This project is for educational purposes and is open for modification and extension.

---

## 👨‍🍳 Made with passion for recipes and ML!
