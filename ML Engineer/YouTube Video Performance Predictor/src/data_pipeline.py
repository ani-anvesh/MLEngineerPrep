import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_and_process_data(filepath):
    df = pd.read_csv(filepath)

    # Feature: Title length
    df['title_length'] = df['title'].apply(lambda x: len(str(x)))

    # Feature: Number of tags
    df['num_tags'] = df['tags'].apply(lambda x: len(str(x).split('|')))

    # Feature: Publish hour
    df['publish_hour'] = pd.to_datetime(df['publish_time']).dt.hour

    # Convert booleans to int
    df['comments_disabled'] = df['comments_disabled'].astype(int)
    df['ratings_disabled'] = df['ratings_disabled'].astype(int)
    df['video_error_or_removed'] = df['video_error_or_removed'].astype(int)

    # Features selected for training
    feature_cols = ['title_length', 'num_tags', 'category_id', 'publish_hour',
                    'comments_disabled', 'ratings_disabled', 'video_error_or_removed']

    X = df[feature_cols]
    y = df['views']

    return X, y
