import unittest
import os
from src import data_pipeline

class TestDataPipeline(unittest.TestCase):

    def setUp(self):
        # Path to sample dataset (make sure this CSV exists at this path)
        self.test_csv = 'data/USvideos.csv'

    def test_load_and_process_data(self):
        # Check if file exists
        self.assertTrue(os.path.exists(self.test_csv), "Dataset file not found.")

        # Load and process data
        X, y = data_pipeline.load_and_process_data(self.test_csv)

        # Check if features and target are not empty
        self.assertFalse(X.empty, "Feature dataframe is empty.")
        self.assertFalse(y.empty, "Target series is empty.")

        # Check expected columns
        expected_columns = [
            'title_length', 
            'num_tags', 
            'category_id', 
            'publish_hour', 
            'comments_disabled', 
            'ratings_disabled', 
            'video_error_or_removed'
        ]
        self.assertTrue(all(col in X.columns for col in expected_columns), "Missing expected feature columns.")

if __name__ == '__main__':
    unittest.main()
