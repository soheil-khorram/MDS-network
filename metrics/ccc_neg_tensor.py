from metric import Metric
import tensorflow as tf


class CCC_NEG_TENSOR(Metric):
    """class for tensor version of CCC."""

    @staticmethod
    def get_name():
        """Return name of the metric."""
        return 'CCC_NEG_TENSOR'

    @staticmethod
    def compute(y_true, y_pred):
        """Calculate the metric."""
        with tf.name_scope(CCC_NEG_TENSOR.get_name()):
            x = y_true
            y = y_pred
            x_mean = tf.reduce_mean(x)
            y_mean = tf.reduce_mean(y)
            xy_cov = tf.reduce_mean(tf.multiply(x, y)) - (x_mean * y_mean)
            x_var = tf.reduce_mean(tf.multiply(x, x)) - (x_mean * x_mean)
            y_var = tf.reduce_mean(tf.multiply(y, y)) - (y_mean * y_mean)
            m = -2 * xy_cov / (x_var + y_var + (x_mean - y_mean) ** 2)
        return m
