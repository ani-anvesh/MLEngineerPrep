import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

def load_data(path):
    return pd.read_csv(path)

def load_and_clean_data(df):
    # Clean the Category column
    df['Category'] = df['Category'].str.lower()
    df['Category'] = df['Category'].fillna("")

    return df

def clean_category(text):
    text = text.lower()
    text = re.sub(r"[^\w\s,]", "", text)  # remove punctuation except commas
    return text

def add_dummy_popularity(df):
    # Example: add fake popularity scores (0 to 100)
    df['popularity'] = list(range(len(df)))[::-1]
    scaler = MinMaxScaler()
    df['popularity_scaled'] = scaler.fit_transform(df[['popularity']])
    return df

def is_best_time(best_time_str, user_month):
    best_months = [month.strip().lower() for month in best_time_str.split(',')]
    return user_month.lower() in best_months

def compute_tfidf_matrix(df, text_column='Category'):
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(df[text_column])
    return tfidf_matrix, tfidf


def recommend_destinations(user_pref_text, tfidf_matrix, tfidf, df, top_n=5):
    user_vec = tfidf.transform([user_pref_text])
    cos_sim = cosine_similarity(user_vec, tfidf_matrix)
    top_indices = cos_sim[0].argsort()[::-1][:top_n]
    top_scores = cos_sim[0][top_indices]
    
    return df.iloc[top_indices].assign(similarity_score=top_scores)

def add_dummy_popularity(df):
    # Example: add fake popularity scores (0 to 100)
    df['popularity'] = list(range(len(df)))[::-1]
    scaler = MinMaxScaler()
    df['popularity_scaled'] = scaler.fit_transform(df[['popularity']])
    return df

def custom_score(row, weights):
    return (
        weights['similarity'] * row['similarity_score'] +
        weights.get('popularity', 0) * row.get('popularity_scaled', 0)
    )

def apply_custom_scoring(df, weights):
    df['final_score'] = df.apply(lambda row: custom_score(row, weights), axis=1)
    return df.sort_values(by='final_score', ascending=False)

def plot_recommendations(df):
    top = df.head(5)
    plt.figure(figsize=(10, 6))
    plt.barh(top['City'], top['similarity_score'], color='skyblue')
    plt.xlabel("Similarity Score")
    plt.title("Top 5 Destination Recommendations")
    plt.gca().invert_yaxis()
    plt.savefig("outputs/top_5_recommendations.png")
    plt.show()