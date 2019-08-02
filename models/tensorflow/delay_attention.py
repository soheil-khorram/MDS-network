"""Class for a convolutional neural network with delay kernel and attention."""
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
#       5- Using l2-regularization is important for valence but not for arousal
#   Parameters have not been tuned yet.


class DelayAttention(TensorflowModel):
    """Class for a convolution deconvolution network."""
    @staticmethod
    def parse_options(parser):
        """Parse options."""
        TensorflowModel.parse_options(parser)
        parser.add_argument('--lr', type=float, default=0.01)
        parser.add_argument('--nb-epochs', type=int, default=300)
        parser.add_argument('--min-grad', type=float, default=-5.)
        parser.add_argument('--max-grad', type=float, default=5.)
        parser.add_argument('--input-dim', type=int, default=160)
        parser.add_argument('--input-len', type=int, default=7501)
        parser.add_argument('--output-len', type=int, default=7501)
        parser.add_argument('--min-delay', type=float, default=-0.8)
        parser.add_argument('--max-delay', type=float, default=0.0)
        parser.add_argument('--delay-init-method', type=str, default='uniform')
        parser.add_argument('--conv-kernel-len', type=int, default=8)
        parser.add_argument('--conv-channel-num', type=int, default=16)
        parser.add_argument('--conv-layer-num', type=int, default=5)
        parser.add_argument('--conv-l2-reg-weight', type=float, default=0.0)
        parser.add_argument('--delay-num', type=int, default=16)
        parser.add_argument('--sigma', type=float, default=0.005)
        parser.add_argument('--kernel-type', type=str, default='sinc')

    def __init__(self, opts):
        """Construct model."""
        super(DelayAttention, self).__init__(opts)
        self.lr = opts.lr
        self.nb_epochs = opts.nb_epochs
        self.min_grad = opts.min_grad
        self.max_grad = opts.max_grad
        self.batch_size = opts.batch_size
        self.input_dim = opts.input_dim
        self.input_len = opts.input_len
        self.output_len = opts.output_len
        self.min_delay = opts.min_delay
        self.max_delay = opts.max_delay
        self.delay_init_method = opts.delay_init_method
        self.conv_kernel_len = opts.conv_kernel_len
        self.conv_channel_num = opts.conv_channel_num
        self.conv_layer_num = opts.conv_layer_num
        self.delay_num = opts.delay_num
        self.sigma = opts.sigma
        self.conv_l2_reg_weight = opts.conv_l2_reg_weight
        self.kernel_type = opts.kernel_type

    def conv_diff_delay_for_each_dim(self, h, length, channel_num, sigma,
                                     kernel_type, initializer, name):
        with tf.name_scope(name):
            self.mu = tf.get_variable(
                name='mu_' + name,
                shape=[channel_num],
                dtype=tf.float32,
                initializer=initializer,
                regularizer=None,
                trainable=True)
            kernel = []
            for i_in in range(channel_num):
                for i_out in range(channel_num):
                    if i_in != i_out:
                        k = []
                        k = tf.zeros([length], tf.float32)
                        kernel.append(k)
                        continue
                    k = []
                    k = np.linspace(-1, 1, length)
                    k = tf.convert_to_tensor(k, dtype=tf.float32)
                    k = (k - self.mu[i_out]) / sigma
                    if kernel_type == 'sinc':
                        nos = tf.where(tf.abs(k) < 0.00001, tf.ones_like(k),
                                       tf.zeros_like(k))
                        nos = tf.reduce_sum(nos)
                        k = tf.cond(nos < 0.5, lambda: k, lambda: k - 0.00005)
                        k = tf.sin(np.pi * k) / (np.pi * k)
                    if kernel_type == 'gaussian':
                        k = tf.exp(-0.5 * (k ** 2))
                    kernel.append(k)
            kernel = tf.stack(kernel, -1)
            kernel = tf.reshape(kernel, [length, channel_num, channel_num])
            h = tf.nn.conv1d(h, kernel, stride=1, padding='SAME',
                             use_cudnn_on_gpu=None, data_format='NHWC')
        return h

    def conv(self, h, shape, activation, use_bias, kernel_initializer,
             bias_initializer, kernel_l2_reg_weight, bias_l2_reg_weight, name):
        with tf.name_scope(name):
            kernel = tf.get_variable(
                name='kernel_' + name,
                shape=shape,
                dtype=tf.float32,
                initializer=kernel_initializer,
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
                    initializer=bias_initializer,
                    regularizer=None,
                    trainable=True)
                h = tf.nn.bias_add(h, b, data_format='NHWC')
                self.loss = self.loss + bias_l2_reg_weight * tf.nn.l2_loss(b)
            if activation is not None:
                h = activation(h)
            self.loss = self.loss + \
                kernel_l2_reg_weight * tf.nn.l2_loss(kernel)
        return h

    def construct(self):
        normal_init = tf.random_normal_initializer()
        zero_init = tf.zeros_initializer()

        self.input_x = tf.placeholder(tf.float32, [self.batch_size,
                                      self.input_len, self.input_dim],
                                      name="input_x")
        self.input_y = tf.placeholder(tf.float32, [self.batch_size,
                                      self.output_len], name="input_y")

        h0 = self.input_x
        h0 = self.conv(
            h0, [self.conv_kernel_len, self.input_dim, self.conv_channel_num],
            tf.nn.tanh, True, normal_init, zero_init, self.conv_l2_reg_weight,
            self.conv_l2_reg_weight, 'conv_init')
        for i in range(self.conv_layer_num - 1):
            h0 = self.conv(h0, [self.conv_kernel_len, self.conv_channel_num,
                           self.conv_channel_num], tf.nn.tanh, True,
                           normal_init, zero_init, self.conv_l2_reg_weight,
                           self.conv_l2_reg_weight, 'conv' + str(i))

        h1 = self.conv(h0, [self.conv_kernel_len, self.conv_channel_num,
                       self.delay_num], tf.nn.tanh, True, normal_init,
                       zero_init, self.conv_l2_reg_weight,
                       self.conv_l2_reg_weight, 'conv_h1_0')
        h2 = self.conv(h0, [self.conv_kernel_len, self.conv_channel_num,
                       self.delay_num], None, False, zero_init,
                       zero_init, self.conv_l2_reg_weight,
                       self.conv_l2_reg_weight, 'conv_h2_0')
        h2 = tf.nn.softmax(h2, dim=-1)
        h = tf.multiply(h2, h1)

        if self.delay_init_method == 'uniform':
            delay_initializer = tf.random_uniform_initializer(
                minval=self.min_delay, maxval=self.max_delay, dtype=tf.float32)
        else:
            raise NotImplementedError()
        h = self.conv_diff_delay_for_each_dim(
            h, 1024, self.delay_num, self.sigma, self.kernel_type,
            delay_initializer, 'conv_diff_delay_for_each_dim')
        h = self.conv(h, [1, self.delay_num, 1], None, False, normal_init,
                      zero_init, 0, 0, 'final_conv')

        h = tf.reshape(h, [self.batch_size, self.output_len],
                       name='output_reshape')
        self.predicted_y = h
        # calculate negative ccc
        self.loss = self.loss + CCC_NEG_TENSOR.compute(self.input_y,
                                                       self.predicted_y)

    @staticmethod
    def get_test_metrics():
        return [RMSE, CCC]

    @staticmethod
    def get_selection_metric():
        return CCC
