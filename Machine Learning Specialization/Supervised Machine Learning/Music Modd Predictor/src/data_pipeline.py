import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(path):
    """Load dataset from the given path."""
    return pd.read_csv(path)

def clean_data(df):
    """Remove unwanted columns and handle missing data."""
    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', axis=1)
    df = df.dropna()  # Drop rows with any missing values
    df = df.reset_index(drop=True)
    missing_counts = df.isnull().sum()
    print(missing_counts[missing_counts > 0])
    return df

def feature_engineering(df):
    """Placeholder for feature engineering (can be extended)."""
    # No specific feature engineering in original notebook
    return df

def plot_distributions(df, output_dir):
    """Generate and save distribution plots."""    
    # Energy distribution
    df.hist(bins=30, figsize=(12, 10))
    plt.title('Energy Distribution')
    plt.xlabel('Energy')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(output_dir, 'Energy_Distribution.png'))
    plt.close()
    
    # Valence distribution
    sns.histplot(data=df, x='valence', bins=30)
    plt.title("Valence Distribution")
    plt.savefig(os.path.join(output_dir, 'Valence_Distribution.png'))
    plt.close()

    # Time signature count
    sns.countplot(data=df, x='time_signature')
    plt.title("Time Signature")
    plt.savefig(os.path.join(output_dir, 'Time_Signature.png'))
    plt.close()