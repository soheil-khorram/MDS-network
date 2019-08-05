# Author: Soheil Khorram
# License: Simplified BSD

import numpy as np
from metric import Metric


class RMSE(Metric):
    """class for RMSE."""

    @staticmethod
    def get_name():
        """Return name of the metric."""
        return 'RMSE'

    @staticmethod
    def compute(y_true, y_pred):
        """Calculate the metric."""
        return np.sqrt(np.mean(np.square(y_true - y_pred)))
