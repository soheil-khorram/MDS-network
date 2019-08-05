# Author: Soheil Khorram
# License: Simplified BSD

"""Class for a convolutional neural network with sinc kernel."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from models.tensorflow.tensorflow_model import TensorflowModel
from metrics.ccc_neg_tensor import CCC_NEG_TENSOR
from metrics.ccc import CCC
from metrics.rmse import RMSE
import tensorflow as tf
import numpy as np


# Ideas:
#       1- Using sinc
#       2- Using uniform negative initialization
#       3- Using mulyiple layers
#       4- To solve the unstability of the sinc if one of the arguments of the
#          sinc is zero then small variation is applied to it
#   Parameters have not been tuned yet.


class SincKernel(TensorflowModel):
    """Class for a convolution deconvolution network."""
    @staticmethod
    def parse_options(parser):
        """Parse options for SincKernel."""
        TensorflowModel.parse_options(parser)
        parser.add_argument('--lr', type=float, default=0.005)
        parser.add_argument('--nb-epochs', type=int, default=500)
        parser.add_argument('--min-grad', type=float, default=-5.)
        parser.add_argument('--max-grad', type=float, default=5.)
        parser.add_argument('--input-dim', type=int, default=160)
        parser.add_argument('--input-len', type=int, default=7501)
        parser.add_argument('--output-len', type=int, default=7501)
        parser.add_argument('--conv-layer-num', type=int, default=2)
        parser.add_argument('--conv-kernel-num', type=int, default=1)
        parser.add_argument('--conv-kernel-len', type=int, default=512)

    def __init__(self, opts):
        """Construct a ConvPool model."""
        super(SincKernel, self).__init__(opts)
        self.lr = opts.lr
        self.nb_epochs = opts.nb_epochs
        self.min_grad = opts.min_grad
        self.max_grad = opts.max_grad
        self.batch_size = opts.batch_size
        self.input_dim = opts.input_dim
        self.input_len = opts.input_len
        self.output_len = opts.output_len
        self.conv_layer_num = opts.conv_layer_num
        self.conv_kernel_num = opts.conv_kernel_num
        self.conv_kernel_len = opts.conv_kernel_len

    def sinc_conv_ver0(self, h, shape, activation, use_bias, sigma, a, name):
        with tf.name_scope(name):
            length, chin, chout = shape
            self.mu = tf.get_variable(
                name='mu_' + name,
                shape=[chout],
                dtype=tf.float32,
                initializer=tf.random_uniform_initializer(minval=-0.8,
                                                          maxval=0.0,
                                                          dtype=tf.float32),
                regularizer=None,
                trainable=True)
            # sigma = tf.get_variable(
            #     name='sigma_' + name,
            #     shape=[chout],
            #     dtype=tf.float32,
            #     initializer=tf.ones_initializer(),
            #     regularizer=None,
            #     trainable=True)
            # sigma = sigma / 100
            # a = tf.get_variable(
            #     name='a_' + name,
            #     shape=[chin, chout],
            #     dtype=tf.float32,
            #     initializer=tf.random_normal_initializer(),
            #     regularizer=None,
            #     trainable=True)
            self.num_of_small = []
            kernel = []
            for i_in in range(chin):
                for i_out in range(chout):
                    k = []
                    k = np.linspace(-1, 1, length)
                    k = tf.convert_to_tensor(k, dtype=tf.float32)
                    k = (k - self.mu[i_out]) / sigma
                    nos = tf.where(tf.abs(k) < 0.00001, tf.ones_like(k),
                                   tf.zeros_like(k))
                    nos = tf.reduce_sum(nos)
                    self.num_of_small.append(nos)
                    k = tf.cond(nos < 0.5, lambda: k, lambda: k - 0.00005)
                    k = a * tf.sin(np.pi * k) / (np.pi * k)
                    kernel.append(k)
            kernel = tf.stack(kernel, -1)
            kernel = tf.reshape(kernel, shape)
            h = tf.nn.conv1d(
                h, kernel, stride=1,
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

    def sinc_conv(self, h, shape, activation, use_bias, sigma, a, name):
        with tf.name_scope(name):
            length, chin, chout = shape
            self.mu = tf.get_variable(
                name='mu_' + name,
                shape=[chout],
                dtype=tf.float32,
                initializer=tf.random_uniform_initializer(minval=-0.8,
                                                          maxval=0.0,
                                                          dtype=tf.float32),
                regularizer=None,
                trainable=True)
            # sigma = tf.get_variable(
            #     name='sigma_' + name,
            #     shape=[chout],
            #     dtype=tf.float32,
            #     initializer=tf.ones_initializer(),
            #     regularizer=None,
            #     trainable=True)
            # sigma = sigma / 100
            # a = tf.get_variable(
            #     name='a_' + name,
            #     shape=[chin, chout],
            #     dtype=tf.float32,
            #     initializer=tf.random_normal_initializer(),
            #     regularizer=None,
            #     trainable=True)
            self.num_of_small = []
            kernel = []
            for i_in in range(chin):
                for i_out in range(chout):
                    k = []
                    k = np.linspace(-1, 1, length)
                    k = tf.convert_to_tensor(k, dtype=tf.float32)
                    k = (k - self.mu[i_out]) / sigma
                    nos = tf.where(tf.abs(k) < 0.00001, tf.ones_like(k),
                                   tf.zeros_like(k))
                    nos = tf.reduce_sum(nos)
                    self.num_of_small.append(nos)
                    k = tf.cond(nos < 0.5, lambda: k, lambda: k - 0.00005)
                    k = a * tf.sin(np.pi * k) / (np.pi * k)
                    kernel.append(k)
            kernel = tf.stack(kernel, -1)
            kernel = tf.reshape(kernel, shape)
            h = tf.nn.conv1d(
                h, kernel, stride=1,
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

    def construct(self):
        self.input_x = tf.placeholder(tf.float32, [self.batch_size,
                                      self.input_len, self.input_dim],
                                      name="input_x")
        self.input_y = tf.placeholder(tf.float32, [self.batch_size,
                                      self.output_len], name="input_y")
        h0 = self.input_x
        h0 = self.conv(h0, [10, 160, 16], tf.nn.tanh, True, 'conv124')
        h0 = self.conv(h0, [10, 16, 16], tf.nn.tanh, True, 'corr24')
        h0 = self.conv(h0, [10, 16, 16], tf.nn.tanh, True, 'consa4')
        h0 = self.conv(h0, [10, 16, 16], tf.nn.tanh, True, 'cnssfsdf')
        h0 = self.conv(h0, [10, 16, 16], tf.nn.tanh, True, 'cnssfssdf')
        h0 = self.conv(h0, [10, 16, 1], tf.nn.tanh, True, 'coer4')
        h0 = self.sinc_conv(h0, shape=[1024, 1, 16], activation=None,
                            use_bias=False, sigma=0.01, a=1, name='gconv12')
        h = h0
        h = self.conv(h, [1, 16, 1], None, False, 'conv12312')
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
