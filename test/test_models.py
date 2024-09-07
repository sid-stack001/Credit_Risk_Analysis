import unittest
from model.random_forest_model import train_random_forest

class TestModels(unittest.TestCase):
    def test_random_forest(self):
        # Dummy data for testing
        X_train = [[0, 1], [1, 0], [1, 1], [0, 0]]
        y_train = [0, 1, 1, 0]
        X_test = [[0, 1], [1, 0]]
        y_test = [0, 1]
        
        model, accuracy = train_random_forest(X_train, y_train, X_test, y_test)
        self.assertGreater(accuracy, 0.5)
