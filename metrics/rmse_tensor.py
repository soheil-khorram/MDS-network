from metric import Metric
import tensorflow as tf


class RMSE_TENSOR(Metric):
    """class for tensor version of RMSE."""

    @staticmethod
    def get_name():
        """Return name of the metric."""
        return 'RMSE_TENSOR'

    @staticmethod
    def compute(y_true, y_pred):
        """Calculate the metric."""
        with tf.name_scope(RMSE_TENSOR.get_name()):
            m = tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))
        return m
