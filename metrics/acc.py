from sklearn.metrics import accuracy_score
from metric import Metric
import numpy as np


class ACC(Metric):
    """class for Acc."""

    @staticmethod
    def get_name():
        """Return name of the metric."""
        return 'ACC'

    @staticmethod
    def compute(y_true, y_pred):
        """Calculate the metric."""
        y_true = np.argmax(y_true, axis=1)
        y_pred = np.argmax(y_pred, axis=1)
        return accuracy_score(y_true, y_pred)
