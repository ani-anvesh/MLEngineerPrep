import nltk
import re
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
# from nltk.tokenize import tab_tokenize

# Download to local folder once
nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Add local folder to NLTK data search path
# nltk.data.path.append('./nltk_data')
# nltk.data.path.append('/Users/anveshradharapu/Library/Caches/pypoetry/virtualenvs/mlengineerprep-fSSVlCLJ-py3.11/nltk_data')

stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

def preprocess_df(df):
    df["clean_text"] = df["text"].apply(clean_text)
    return df


def load_data(csv_path):
    try:
        df = pd.read_csv(csv_path)
        if 'text' not in df.columns or 'label' not in df.columns:
            raise ValueError("CSV must have 'text' and 'label' columns.")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None
    
