# climate-stance-project/src/data_pipeline.py

import pandas as pd
import re
import string
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

def load_dataset(filepath):
    return pd.read_csv(filepath)

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)  # remove URLs
    text = re.sub(f'[{string.punctuation}]', '', text)  # remove punctuation
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words and word.isalpha()]
    return ' '.join(tokens)

def preprocess_dataframe(df, text_column='tweet'): 
    df[text_column] = df[text_column].astype(str).apply(clean_text)
    return df

def vectorize_text(corpus):
    tfidf = TfidfVectorizer()
    X = tfidf.fit_transform(corpus)
    return X, tfidf

def split_data(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42)

def evaluate_model(y_true, y_pred, model_name):
    print(f"\n{model_name} Evaluation Report:")
    print(classification_report(y_true, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
