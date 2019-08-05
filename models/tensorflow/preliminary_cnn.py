# Author: Soheil Khorram
# License: Simplified BSD

"""Class for the preliminary experiment."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from models.tensorflow.tensorflow_model import TensorflowModel
from metrics.ccc_neg_tensor import CCC_NEG_TENSOR
from metrics.ccc import CCC
from metrics.rmse import RMSE
import tensorflow as tf
import numpy as np


class PreliminaryCnn(TensorflowModel):
    """Class for the preliminary experiment."""
    @staticmethod
    def parse_options(parser):
        """Parse options for SincKernel."""
        TensorflowModel.parse_options(parser)
        parser.add_argument('--lr', type=float, default=0.005)
        parser.add_argument('--nb-epochs', type=int, default=200)
        parser.add_argument('--min-grad', type=float, default=-5.)
        parser.add_argument('--max-grad', type=float, default=5.)
        parser.add_argument('--input-dim', type=int, default=160)
        parser.add_argument('--input-len', type=int, default=7501)
        parser.add_argument('--output-len', type=int, default=7501)
        parser.add_argument('--kernel-len', type=int, default=32)
        parser.add_argument('--frame-delay', type=int, default=64)

    def __init__(self, opts):
        """Initialize the class."""
        super(PreliminaryCnn, self).__init__(opts)
        self.lr = opts.lr
        self.nb_epochs = opts.nb_epochs
        self.min_grad = opts.min_grad
        self.max_grad = opts.max_grad
        self.batch_size = opts.batch_size
        self.input_dim = opts.input_dim
        self.input_len = opts.input_len
        self.output_len = opts.output_len
        self.kernel_len = opts.kernel_len
        self.frame_delay = opts.frame_delay

    def conv(self, h, shape, activation, use_bias, name):
        with tf.name_scope(name):
            kernel = tf.get_variable(
                name='kernel_' + name,
                shape=shape,
                dtype=tf.float32,
                initializer=tf.random_normal_initializer(),
                regularizer=None,
                trainable=True)
            h = tf.nn.conv1d(
                h,
                kernel,
                stride=1,
                padding='SAME',
                use_cudnn_on_gpu=None,
                data_format='NHWC')
            if use_bias:
                b = tf.get_variable(
                    name='bias_' + name,
                    shape=[shape[-1]],
                    dtype=tf.float32,
                    initializer=tf.zeros_initializer(),
                    regularizer=None,
                    trainable=True)
                h = tf.nn.bias_add(h, b, data_format='NHWC')
            if activation is not None:
                h = activation(h)
        return h

    def conv_delay(self, h, frame_delay, name):
        with tf.name_scope(name):
            kernel_np = np.zeros(shape=(2 * frame_delay + 1), dtype=np.float32)
            kernel_np[0] = 1
            kernel_tf = tf.stack(kernel_np)
            kernel_tf = tf.reshape(kernel_tf, [2 * frame_delay + 1, 1, 1],
                                   name='kernel_tf_reshape')
            h = tf.nn.conv1d(
                h,
                kernel_tf,
                stride=1,
                padding='SAME',
                use_cudnn_on_gpu=None,
                data_format='NHWC')
        return h

    def construct(self):
        self.input_x = tf.placeholder(tf.float32, [self.batch_size,
                                      self.input_len, self.input_dim],
                                      name="input_x")
        self.input_y = tf.placeholder(tf.float32, [self.batch_size,
                                      self.output_len], name="input_y")
        h0 = self.input_x
        h0 = self.conv(h0, [self.kernel_len, 160, 1], tf.nn.tanh, True,
                       'conv1')
        h0 = self.conv(h0, [1, 1, 1], None, False, 'conv2')
        h0 = self.conv_delay(h0, self.frame_delay, 'conv_delay')
        h = h0
        h = tf.reshape(h, [self.batch_size, self.output_len],
                       name='output_reshape')
        self.predicted_y = h
        # calculate negative ccc
        self.loss = CCC_NEG_TENSOR.compute(self.input_y, self.predicted_y)

    @staticmethod
    def get_test_metrics():
        return [RMSE, CCC]

    @staticmethod
    def get_selection_metric():
        return CCC
