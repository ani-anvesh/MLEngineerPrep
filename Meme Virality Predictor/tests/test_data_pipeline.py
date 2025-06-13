import pandas as pd
from src import data_pipeline

def test_load_data():
    df = data_pipeline.load_data('tests/sample_music.csv')
    assert isinstance(df, pd.DataFrame)

def test_clean_data_removes_column():
    df = pd.DataFrame({
        'Unnamed: 0': [1, 2, 3],
        'valence': [0.5, 0.7, 0.2],
        'time_signature': [4, 3, 4]
    })
    cleaned = data_pipeline.clean_data(df)
    assert 'Unnamed: 0' not in cleaned.columns
    assert not cleaned.isnull().any().any()