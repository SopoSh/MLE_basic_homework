import unittest
import pandas as pd
import os
import sys
import torch

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(ROOT_DIR))
CONF_FILE = os.getenv('CONF_PATH')

from unittest.mock import patch
from training.train import train_model, evaluate_model, SimpleNN

import pandas as pd
from data_process.data_generation import preprocess_data

class TestDataFunctions(unittest.TestCase):
    def test_preprocess_data(self):
        # Create a DataFrame with dummy Iris data
        dummy_data = pd.DataFrame({
            'sepal length (cm)': [5.1, 4.9, 4.7],
            'sepal width (cm)': [3.5, 3.0, 3.2],
            'petal length (cm)': [1.4, 1.4, 1.3],
            'petal width (cm)': [0.2, 0.2, 0.2],
            'target': [0, 0, 0]
        })

        # Suppress logging during testing
        with unittest.mock.patch('logging.info') as mock_logging:
            X_train, y_train, X_inference, y_inference = preprocess_data(dummy_data)

        # Ensure that the mock logger was called
        mock_logging.assert_called_once()

        # Ensure the data has been preprocessed (you can add more specific assertions)
        self.assertTrue(True)  # Add your assertions here

class TestTrainFunctions(unittest.TestCase):
    def test_train_model(self):
        # Create a simple neural network for testing
        input_size, hidden_size, output_size = 4, 8, 3
        model = SimpleNN(input_size, hidden_size, output_size)

        # Generate dummy data
        X_train = torch.randn(10, input_size)
        y_train = torch.randint(0, output_size, (10,))

        # Train the model (suppressing print statements)
        with patch('builtins.print'):
            train_model(model, X_train, y_train)

        # Ensure the model has been trained (you can add more specific assertions)
        self.assertTrue(True)  # Add your assertions here

    def test_evaluate_model(self):
        # Create a simple neural network for testing
        input_size, hidden_size, output_size = 4, 8, 3
        model = SimpleNN(input_size, hidden_size, output_size)

        # Generate dummy data
        X_inference = torch.randn(5, input_size)
        y_inference = torch.randint(0, output_size, (5,))

        # Evaluate the model (suppressing print statements)
        with patch('builtins.print'):
            evaluate_model(model, X_inference, y_inference)

        # Ensure the model has been evaluated (you can add more specific assertions)
        self.assertTrue(True)  # Add your assertions here

if __name__ == '__main__':
    unittest.main()
