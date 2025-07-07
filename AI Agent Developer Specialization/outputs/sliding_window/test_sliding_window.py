import unittest

from your_module import sliding_window  # Replace 'your_module' with the actual module name

class TestSlidingWindow(unittest.TestCase):
    def test_basic_functionality(self):
        self.assertEqual(sliding_window([1, 3, -1, -3, 5, 3, 6, -1, -3], 3), [7, 6, 12, 15, 18, 12, 15, 18, 12])
    
    def test_empty_list(self):
        self.assertEqual(sliding_window([], 3), [])

    def test_edge_case_smaller_list(self):
        self.assertEqual(sliding_window([1], 3), [])

    def test_edge_case_larger_list(self):
        self.assertEqual(sliding_window([1, 2, 3], 4), [1, 2, 3])

    def test_error_case_negative_numbers(self):
        with self.assertRaises(ValueError):
            sliding_window([1, 2, 3], 4) 

if __name__ == '__main__':
    unittest.main()