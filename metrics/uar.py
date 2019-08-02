from sklearn.metrics import recall_score
from metric import Metric
import numpy as np


class UAR(Metric):
    """class for UAR."""

    @staticmethod
    def get_name():
        """Return name of the metric."""
        return 'UAR'

    @staticmethod
    def compute(y_true, y_pred):
        """Calculate the metric."""
        y_true = np.argmax(y_true, axis=1)
        y_pred = np.argmax(y_pred, axis=1)
        uar = recall_score(y_true, y_pred, average='macro')
        return uar
