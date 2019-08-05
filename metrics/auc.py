# Author: Soheil Khorram
# License: Simplified BSD

from sklearn.metrics import roc_auc_score
from metric import Metric
import numpy as np


class AUC(Metric):
    """class for AUC."""

    @staticmethod
    def get_name():
        """Return name of the metric."""
        return 'AUC'

    @staticmethod
    def compute(y_true, y_pred):
        """Calculate the metric."""
        y_true = np.argmax(y_true, axis=1)
        y_pred = np.argmax(y_pred, axis=1)
        auc = roc_auc_score(y_true, y_pred)
        return auc
