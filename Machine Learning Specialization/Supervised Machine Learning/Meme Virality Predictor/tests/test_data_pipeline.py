import pandas as pd
from src import data_pipeline

def test_load_data():
    df = data_pipeline.load_data('tests/sample_data.csv')
    assert isinstance(df, pd.DataFrame)

def test_feature_engineering():
    sample = pd.DataFrame({'Upvotes': [10, 500, 1500], 'Title': ['A', 'Test', 'Meme']})
    result = data_pipeline.feature_engineering(sample)
    assert 'log_upvotes' in result.columns
    assert 'caption_length' in result.columns
    assert 'virality' in result.columns
    assert result['virality'].tolist() == ['low', 'medium', 'high']

    