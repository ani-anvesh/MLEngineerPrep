import pandas as pd
import numpy as np
import os

def load_data(path):
    """Loads dataset from the given path."""
    return pd.read_csv(path)

def clean_data(df):
    """Cleans the dataset. Placeholder for missing value handling."""
    print(df.isnull().sum())
    df = df.dropna(subset=['Title', 'Upvotes'])  # Keep rows with valid data
    df = df.reset_index(drop=True) 
    return df

def label_virality(upvotes):
    """Labels virality based on upvotes."""
    if upvotes < 100:
        return 'low'
    elif upvotes < 1000:
        return 'medium'
    else:
        return 'high'

def feature_engineering(df):
    """Applies feature engineering to the dataset."""
    df['log_upvotes'] = np.log1p(df['Upvotes'])
    df['caption_length'] = df['Title'].apply(lambda x: len(str(x)))
    df['virality'] = df['Upvotes'].apply(label_virality)
    df['log_caption_length'] = np.log1p(df['caption_length'])
    return df