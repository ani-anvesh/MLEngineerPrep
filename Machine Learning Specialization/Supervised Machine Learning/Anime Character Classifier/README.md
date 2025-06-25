# 🌸 Anime Character Classifier

A machine learning pipeline that classifies anime characters into **Top Loved**, **Top Hated**, or **Neutral** categories using attributes like appearance, tags, and popularity metrics.

---

## 🎯 Objective

To build a logistic regression model that predicts public sentiment toward anime characters using structured data and embedded personality traits.

---

## 📁 Dataset

- **Size:** 132,028 characters
- **Features:**
  - `gender`, `hair_color`
  - `love_rank`, `hate_rank`
  - `love_count`, `hate_count`
  - `tags` (multi-label attributes like "Detectives", "Shy", "Martial Artists")

---

## 🛠️ Pipeline Overview

| Stage               | Description                                                                 |
|---------------------|-----------------------------------------------------------------------------|
| 🧼 Data Cleaning     | Handled missing values, standardized tags, converted counts to numeric     |
| 🧩 Feature Engineering | Encoded categorical fields, binarized tags, and reduced tag features via PCA |
| 🧠 Label Creation    | Categorized characters using quantile thresholds on love/hate rankings and counts |
| 🤖 Model             | Multinomial Logistic Regression (Softmax) using scikit-learn                |
| 📊 Evaluation        | Precision, Recall, F1-score, Confusion Matrix, ROC-AUC                      |

---

## 📊 Model Performance

| Label       | Precision | Recall | F1-Score | Support |
|-------------|-----------|--------|----------|---------|
| Neutral     | **1.00**  | **1.00** | **1.00** | 18,880  |
| Top Hated   | **0.99**  | **0.99** | **0.99** | 2,523   |
| Top Loved   | **0.99**  | **1.00** | **0.99** | 5,003   |
| **Overall Accuracy** |   |   | **1.00** | 26,406   |

- **Macro Average F1:** 0.99  
- **Weighted Average F1:** 1.00  
- 📈 ROC-AUC also indicates excellent separability (optional)

---

## 🧪 Evaluation Module

Custom metrics are modularized in [`src/evaluation.py`](src/evaluation.py), including:

- `evaluate_classification()`
- `evaluate_roc_auc()`

Tested with edge cases in `tests/test_evaluation.py`.

---

## 📌 Key Insights

PCA reduced ~200+ tags to 50 informative dimensions

Top Loved/Hated characters strongly correlated with specific tags (e.g., “Hot-Headed”, “Glasses”)

Minimal overfitting due to simple linear model + dimensionality control

## 🔮 Future Enhancements

Add sampling strategies to balance classes

Explore non-linear models like Random Forest or XGBoost

Use word embeddings or transformer-based encoders for richer tag representations

## 🚀 Usage

```bash
# Train model
python train.py

# Run evaluation
python evaluate.py

---

## ✅ Testing

This project includes unit tests for key modules like `evaluation.py` and `data_pipeline.py`.

### 📂 Test Structure
Tests are located in the `/tests` directory and cover:
- Metric evaluation logic (accuracy, precision, recall, F1)
- Data loading
- Edge cases (imbalanced data, invalid input, perfect/zero predictions)

### 🚀 Run Tests

1. Navigate to the project root:

```bash
cd path/to/anime-character-classifier```

2. run command:

```bash 
pytest```

2. run command:

```bash 
PYTHONPATH=. pytest```