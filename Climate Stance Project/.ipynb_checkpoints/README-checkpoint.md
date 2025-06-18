# 🌍 Climate Change Tweet Stance Classifier

A Natural Language Processing pipeline that classifies tweets related to climate change into one of three categories: **Believer**, **Denier**, or **Neutral**. The project leverages machine learning algorithms and TF-IDF feature extraction to model public stance using tweet text.

---

## 🎯 Objective

To accurately identify the stance of users on climate change using their tweets by building interpretable models like Decision Tree, Random Forest, and XGBoost.

---

## 📁 Dataset

- **Source:** [Kaggle - Climate Change Tweet Dataset](https://www.kaggle.com/)
- **Size:** ~10,000 tweets
- **Target Labels:**
  - `believer` – Supports climate change action
  - `denier` – Disputes or rejects climate change
  - `neutral` – No clear stance

---

## 🛠️ Pipeline Overview

| Stage                | Description                                                                 |
|----------------------|-----------------------------------------------------------------------------|
| 🧼 Preprocessing      | Cleaned text, removed stopwords, tokenized and lemmatized                   |
| 🔍 Feature Extraction | TF-IDF vectorization with unigrams and bigrams                             |
| 🧠 Model Training     | Used Decision Tree, Random Forest, and XGBoost classifiers                  |
| 📊 Evaluation         | Reports accuracy, precision, recall, F1-score, confusion matrices, ROC-AUC  |

---

## ⚙️ Model Performance

### ✅ Decision Tree

- **CV Accuracy:** 0.6556 ± 0.0053  
- **Test Accuracy:** 0.66

| Class     | Precision | Recall | F1-Score |
|-----------|-----------|--------|----------|
| Believer  | 0.80      | 0.79   | 0.79     |
| Denier    | 0.21      | 0.21   | 0.21     |
| Neutral   | 0.30      | 0.31   | 0.31     |

---

### ✅ Random Forest

- **CV Accuracy:** 0.7498 ± 0.0033  
- **Test Accuracy:** 0.75

| Class     | Precision | Recall | F1-Score |
|-----------|-----------|--------|----------|
| Believer  | 0.78      | 0.95   | 0.86     |
| Denier    | 0.34      | 0.07   | 0.12     |
| Neutral   | 0.49      | 0.21   | 0.29     |

---

### ✅ XGBoost

- **CV Accuracy:** 0.7601 ± 0.0026  
- **Test Accuracy:** 0.76

| Class     | Precision | Recall | F1-Score |
|-----------|-----------|--------|----------|
| Believer  | 0.78      | 0.97   | 0.86     |
| Denier    | 0.44      | 0.05   | 0.09     |
| Neutral   | 0.58      | 0.20   | 0.29     |

---

## 🧪 Evaluation Module

Custom evaluation utilities are modularized in [`src/evaluation.py`](src/evaluation.py):

- `evaluate_classification()` – Computes precision, recall, F1
- `evaluate_roc_auc()` – Calculates ROC-AUC
- `save_classification_reports()` – Saves formatted reports

---

## 📌 Key Observations

- **XGBoost** showed best overall performance, particularly for the dominant *believer* class.
- **Denier** tweets remain hardest to classify across all models—indicates potential data imbalance or linguistic ambiguity.
- **Neutral** tweets overlap semantically with both extremes, challenging model separability.

---

## 🔮 Future Enhancements

- Incorporate **transformer-based models** like BERT for contextual understanding.
- Use **SMOTE or class weights** to address imbalance for underrepresented classes.
- Perform **topic modeling** to understand the discourse dimensions driving classification.

---

## Run all tests
pytest

# Or with module path setup
PYTHONPATH=. pytest

