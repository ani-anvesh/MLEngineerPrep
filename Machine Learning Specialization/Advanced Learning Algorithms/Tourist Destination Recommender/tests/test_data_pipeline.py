import unittest
import pandas as pd
from project.src.data_pipeline import preprocess_text

class TestDataPipeline(unittest.TestCase):
    def test_preprocess_text(self):
        df = pd.DataFrame({"description": ["Hello World!", "Test 123."]})
        df_clean = preprocess_text(df)
        self.assertEqual(df_clean.iloc[0]['description'], "hello world")
