import pandas as pd
import numpy as np
import os
import ast

def load_data(path):
    """Loads dataset from the given path."""
    return pd.read_csv(path)

def clean_columns(df):
    """Cleans the column names."""
    df.columns = (
    df.columns
    .str.strip()             # Remove leading/trailing spaces
    .str.lower()             # Convert to lowercase
    .str.replace(' ', '_')   # Replace spaces with underscores
    .str.replace(r'[^\w_]', '', regex=True)  # Remove non-word characters
    )
    print(df.columns)
    return df


def clean_data(df):
    """Cleans the dataset. Handles missing values, parses tags, and removes unused columns."""
    df.drop(columns=['description', 'url', 'eye_color', 'birthday', 'blood_type', 'alias'], inplace=True, errors='ignore')
    
    # Print nulls for initial inspection
    print("Missing values before cleaning:\n", df.isnull().sum())

    # Replace 'unknown' strings with np.nan (for object columns only)
    df = df.apply(lambda col: col.replace('Unknown', np.nan))
    df = df.apply(lambda col: col.fillna(col.mode()[0]) if col.isnull().any() else col)

    # Convert tag strings into lists
    df['tags'] = df['tags'].apply(lambda x: [tag.strip() for tag in str(x).split(',')])
    
    # Optional: Clean column names (make them ML-friendly)
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(' ', '_')
        .str.replace(r'[^\w_]', '', regex=True)
    )

    df.dropna(subset=['gender', 'hair_color', 'love_rank', 'hate_rank'], inplace=True)
    df['love_count'] = pd.to_numeric(df['love_count'], errors='coerce')
    df['hate_count'] = pd.to_numeric(df['hate_count'], errors='coerce')
    df[['love_count', 'hate_count']] = df[['love_count', 'hate_count']].fillna(0).astype(int)
    df['gender'] = df['gender'].str.lower().str.strip()
    df['hair_color'] = df['hair_color'].str.lower().str.strip()

    return df

def count_unknowns(df):
    for col in df.columns:
        if df[col].dtype == 'object':
            unknown_count = (df[col].str.lower() == 'unknown').sum()
            if unknown_count > 0:
                print(f"{col}: {unknown_count} unknowns")

def label_character_factory(love_rk, love_ct, hate_rk, hate_ct):
    def label_func(row):
        if row['love_rank'] <= love_rk and row['love_count'] >= love_ct:
            return 'Top Loved'
        elif row['hate_rank'] <= hate_rk and row['hate_count'] >= hate_ct:
            return 'Top Hated'
        else:
            return 'Neutral'
    return label_func