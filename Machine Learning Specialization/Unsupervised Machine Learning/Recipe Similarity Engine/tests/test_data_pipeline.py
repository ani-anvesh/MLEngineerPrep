import pandas as pd
from src.data_pipeline import preprocess_ingredients

def test_preprocessing():
    df = pd.DataFrame({'ingredients': [['egg', 'milk'], ['beef', 'onion']]})
    processed = preprocess_ingredients(df)
    assert 'joined_ingredients' in processed.columns
    assert processed.shape[0] == 2
