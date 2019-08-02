from metric import Metric


class CCC(Metric):
    """class for CCC."""

    @staticmethod
    def get_name():
        """Return name of the metric."""
        return 'CCC'

    @staticmethod
    def compute(y_true, y_pred):
        """Calculate the metric."""
        x = y_true
        y = y_pred
        xMean = x.mean()
        yMean = y.mean()
        xyCov = (x * y).mean() - (xMean * yMean)
        xVar = x.var()
        yVar = y.var()
        return 2 * xyCov / (xVar + yVar + (xMean - yMean) ** 2)
