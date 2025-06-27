# Mental Health Journal Sentiment Analyzer

## Overview

This project builds a text classification model to analyze emotional tone in mental health journal entries. It classifies entries into five emotion categories:

- Sadness (0)
- Joy (1)
- Love (2)
- Anger (3)
- Fear (4)

The model is trained using Logistic Regression on TF-IDF features extracted from preprocessed text data.

---

## Project Structure

```
project/
├── data/               # Raw and processed datasets (e.g. emotion_dataset.csv)
├── models/             # Saved models and vectorizers
├── notebooks/          # Jupyter notebooks for exploration and training
├── outputs/            # Evaluation reports, plots (e.g. confusion matrix)
├── src/                # Source code for preprocessing and evaluation
│   ├── data_pipeline.py    # Text preprocessing functions
│   └── evaluation.py       # Evaluation and plotting functions
├── tests/              # Unit tests for pipeline
├── README.md           # This file
├── requirements.txt    # Required Python packages
```

---

## Setup Instructions

1. Clone the repository and navigate to the project folder:

```bash
git clone <repo-url>
cd project
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Place your dataset (`emotion_dataset.csv`) in the `data/` folder.

---

## Usage

### 1. Preprocess Data

Run the preprocessing script or use the `preprocess_df` function in `src/data_pipeline.py` to clean the raw text.

### 2. Train Model

Train a Logistic Regression classifier with TF-IDF features using the notebook or training script. Example snippet:

```python
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(train_texts)

clf = LogisticRegression(max_iter=1000)
clf.fit(X_train_vec, train_labels)
```

### 3. Evaluate

Evaluate on test set and generate classification reports and confusion matrices.

Example classification report from one run:

```
              precision    recall  f1-score   support

           0       0.90      0.93      0.92       933
           1       0.84      0.96      0.89      1072
           2       0.86      0.62      0.72       261
           3       0.91      0.83      0.87       432
           4       0.86      0.80      0.83       387
           5       0.89      0.56      0.68       115

    accuracy                           0.87      3200
   macro avg       0.88      0.78      0.82      3200
weighted avg       0.87      0.87      0.87      3200
```

### 4. Visualize Mood Trends

If timestamped journal entries are available, visualize emotion frequency trends over time using `matplotlib` or `seaborn`.

### 5. Save Artifacts

Save trained models and vectorizers for future use:

```python
import joblib

joblib.dump(clf, "models/log_reg_model.pkl")
joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")
```

---

## Running the App

Add a CLI or Streamlit interface to input new journal entries and receive emotion predictions.

Example CLI snippet:

```python
entry = input("Enter your journal text:\n")
cleaned = preprocess_text(entry)
vec = vectorizer.transform([cleaned])
emotion = clf.predict(vec)[0]
print(f"Predicted emotion: {emotion}")
```

---

## Requirements

- pandas
- scikit-learn
- nltk
- matplotlib
- seaborn
- joblib
- streamlit (optional for UI)

---

## License

This project is open-source and free to use.

---

## Contact

For questions or contributions, please open an issue or contact the author.
