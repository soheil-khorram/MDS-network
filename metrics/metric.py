"""Base abstract class for each metric class."""


class Metric(object):
    """Base abstract class for each metric class."""

    @staticmethod
    def get_name():
        """Return name of the metric."""
        pass

    @staticmethod
    def compute(y_true, y_pred):
        """Calculate the metric."""
        pass

    def __hash__(self):
        return hash(self.get_name())

    def __eq__(self, other):
        return self.get_name() == other.get_name()

    def __ne__(self, other):
        return not(self == other)
