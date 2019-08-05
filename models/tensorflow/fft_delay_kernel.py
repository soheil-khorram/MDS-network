# Author: Soheil Khorram
# License: Simplified BSD

"""Class for a convolution deconvolution neural network."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from models.tensorflow_model import TensorflowModel
from metrics.ccc_neg_tensor import CCC_NEG_TENSOR
# from metrics.rmse_tensor import RMSE_TENSOR
import tensorflow as tf
import numpy as np


class FftDelayKernel(TensorflowModel):
    """Class for a convolution deconvolution network."""

    def __init__(self, opts):
        """Construct a ConvPool model."""
        super(FftDelayKernel, self).__init__(opts)
        self.batch_size = opts.batch_size
        self.input_dim = opts.input_dim
        self.input_len = opts.input_len
        self.output_len = opts.output_len
        self.conv_layer_num = opts.conv_layer_num
        self.conv_kernel_num = opts.conv_kernel_num
        self.conv_kernel_len = opts.conv_kernel_len

    @staticmethod
    def parse_options(parser):
        """Parse options for TensorflowConvDeconv."""
        TensorflowModel.parse_options(parser)
        parser.add_argument('--input-dim', type=int, default=160)
        parser.add_argument('--input-len', type=int, default=7501)
        parser.add_argument('--output-len', type=int, default=7501)
        parser.add_argument('--conv-layer-num', type=int, default=2)
        parser.add_argument('--conv-kernel-num', type=int, default=1)
        parser.add_argument('--conv-kernel-len', type=int, default=512)

    def fft_delay_conv(self, h, batch_size, chin, chout, activation, use_bias,
                       name):
        with tf.name_scope(name):
            self.mu = tf.get_variable(
                name='mu_' + name,
                shape=[chout],
                dtype=tf.float32,
                initializer=tf.random_uniform_initializer(minval=-700,
                                                          maxval=0.0,
                                                          dtype=tf.float32),
                regularizer=None,
                trainable=True)
            h_list = []
            for b in range(batch_size):
                for i_out in range(chout):
                    sum_chin = tf.zeros([7501], dtype=tf.float32)
                    for i_in in range(chin):
                        t = h[b, :, i_in]
                        t = tf.cast(t, tf.complex64)
                        t = tf.fft(t)
                        phase = tf.exp(
                            -1j * 2 * np.pi * np.array(range(7501)) *
                            self.mu[i_out] / 7501)
                        phase = tf.cast(phase, tf.complex64)
                        t = t * phase
                        t = tf.ifft(t)
                        t = tf.cast(t, tf.float32)
                        sum_chin = sum_chin + t
                    h_list.append(sum_chin)
            h = tf.stack(h_list, -1)
            h = tf.reshape(h, [7501, batch_size, chout])
            h = tf.transpose(h, perm=[1, 0, 2])
            if use_bias:
                b = tf.get_variable(
                    name='bias_' + name,
                    shape=[chout],
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
            #     trainable=False)
            # sigma = sigma / 10
            # a = tf.get_variable(
            #     name='a_' + name,
            #     shape=[chin, chout],
            #     dtype=tf.float32,
            #     initializer=tf.random_normal_initializer(),
            #     regularizer=None,
            #     trainable=True)
            kernel = []
            for i_in in range(chin):
                for i_out in range(chout):
                    k = []
                    k = np.linspace(-1, 1, length)
                    k = tf.convert_to_tensor(k, dtype=tf.float32)
                    k = (k - self.mu[i_out]) / sigma
                    k = a * tf.sin(np.pi * k) / (np.pi * k)
                    # k = tf.nn.tanh(k)
                    # k = tf.clip_by_value(k, clip_value_min=-1.001*a,
                    #                      clip_value_max=1.001*a)
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
        h0 = self.fft_delay_conv(h0, self.batch_size, 1, 16,
                                 activation=tf.nn.tanh, use_bias=False,
                                 name='fft_delay_conv2134')
        h = h0
        h = self.conv(h, [1, 16, 1], None, False, 'conv12312')
        h = tf.reshape(h, [self.batch_size, self.output_len],
                       name='output_reshape')
        self.predicted_y = h
        # calculate negative ccc
        self.loss = CCC_NEG_TENSOR.compute(self.input_y, self.predicted_y)
        # self.loss = RMSE_TENSOR.compute(self.input_y, self.predicted_y)
