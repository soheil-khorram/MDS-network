"""Class for a convolution deconvolution neural network."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from models.tensorflow.tensorflow_model import TensorflowModel
from metrics.ccc_neg_tensor import CCC_NEG_TENSOR
from metrics.ccc import CCC
from metrics.rmse import RMSE
import tensorflow as tf

l2 = tf.contrib.layers.l2_regularizer


class ConvDeconv(TensorflowModel):
    """Class for a convolution deconvolution network."""
    @staticmethod
    def parse_options(parser):
        """Parse options for ConvDeconv."""
        TensorflowModel.parse_options(parser)
        parser.add_argument('--lr', type=float, default=0.0001)
        parser.add_argument('--nb-epochs', type=int, default=300)
        parser.add_argument('--min-grad', type=float, default=-5.)
        parser.add_argument('--max-grad', type=float, default=5.)
        parser.add_argument('--input-dim', type=int, default=160)
        parser.add_argument('--input-len', type=int, default=7501)
        parser.add_argument('--output-len', type=int, default=7501)
        parser.add_argument('--subsample-rate', type=int, default=2)
        parser.add_argument('--nb-conv-layers', type=int, default=7)
        parser.add_argument('--nb-conv-kernels', type=int, default=64)
        parser.add_argument('--conv-kernel-size', type=int, default=4)
        parser.add_argument('--conv-l2-reg', type=float, default=.04)
        parser.add_argument('--nb-inter-conv-layers', type=int, default=1)
        parser.add_argument('--nb-inter-conv-kernels', type=int, default=64)
        parser.add_argument('--inter-conv-kernel-size', type=int, default=1)
        parser.add_argument('--inter-conv-l2-reg', type=float, default=.04)
        parser.add_argument('--nb-deconv-layers', type=int, default=7)
        parser.add_argument('--nb-deconv-kernels', type=int, default=64)
        parser.add_argument('--deconv-kernel-size', type=int, default=4)
        parser.add_argument('--deconv-l2-reg', type=float, default=.04)

    def __init__(self, opts):
        """Construct a ConvPool model."""
        super(ConvDeconv, self).__init__(opts)
        self.lr = opts.lr
        self.nb_epochs = opts.nb_epochs
        self.min_grad = opts.min_grad
        self.max_grad = opts.max_grad
        self.batch_size = opts.batch_size
        self.input_dim = opts.input_dim
        self.input_len = opts.input_len
        self.output_len = opts.output_len
        self.nb_conv_layers = opts.nb_conv_layers
        self.nb_conv_kernels = opts.nb_conv_kernels
        self.conv_kernel_size = opts.conv_kernel_size
        self.conv_l2_reg = opts.conv_l2_reg
        self.subsample_rate = opts.subsample_rate
        self.nb_inter_conv_layers = opts.nb_inter_conv_layers
        self.nb_inter_conv_kernels = opts.nb_inter_conv_kernels
        self.inter_conv_kernel_size = opts.inter_conv_kernel_size
        self.inter_conv_l2_reg = opts.inter_conv_l2_reg
        self.nb_deconv_layers = opts.nb_deconv_layers
        self.nb_deconv_kernels = opts.nb_deconv_kernels
        self.deconv_kernel_size = opts.deconv_kernel_size
        self.deconv_l2_reg = opts.deconv_l2_reg

    def _crop(self, h):
        """Crop an input tensor."""
        with tf.name_scope("crop"):
            h_len = h.get_shape()[2]
            diff = h_len - self.output_len
            left_ind = int((diff.value + 1) / 2.0)
            h = h[:, :, left_ind:left_ind + self.output_len, :]
        return h

    def construct(self):
        """Construct a ConvPool layer."""
        self.input_x = tf.placeholder(tf.float32, [self.batch_size,
                                      self.input_len, self.input_dim],
                                      name="input_x")
        self.input_y = tf.placeholder(tf.float32, [self.batch_size,
                                      self.output_len], name="input_y")
        xavier_init = tf.contrib.layers.xavier_initializer()
        zero_init = tf.zeros_initializer()
        h = self.input_x
        # reshape
        h = tf.reshape(h, [self.batch_size, 1, self.input_len, self.input_dim],
                       name='input_reshape')
        # conv + max pool layers
        for i in range(self.nb_conv_layers):
            h = tf.layers.conv2d(
                inputs=h,
                filters=self.nb_conv_kernels,
                kernel_size=(1, self.conv_kernel_size),
                strides=(1, 1),
                padding='same',
                data_format='channels_last',
                dilation_rate=(1, 1),
                activation=tf.nn.tanh,
                use_bias=True,
                kernel_initializer=xavier_init,
                bias_initializer=zero_init,
                kernel_regularizer=l2(self.conv_l2_reg),
                bias_regularizer=l2(self.conv_l2_reg),
                activity_regularizer=None,
                trainable=True,
                name='conv' + str(i),
                reuse=None)
            h = tf.nn.max_pool(h, ksize=[1, 1, self.subsample_rate, 1],
                               strides=[1, 1, self.subsample_rate, 1],
                               padding='SAME', data_format='NHWC',
                               name='maxpool' + str(i))
        # middle conv layers
        for i in range(self.nb_inter_conv_layers):
            h = tf.layers.conv2d(
                inputs=h,
                filters=self.nb_inter_conv_kernels,
                kernel_size=(1, self.inter_conv_kernel_size),
                strides=(1, 1),
                padding='same',
                data_format='channels_last',
                dilation_rate=(1, 1),
                activation=tf.nn.tanh,
                use_bias=True,
                kernel_initializer=xavier_init,
                bias_initializer=zero_init,
                kernel_regularizer=l2(self.inter_conv_l2_reg),
                bias_regularizer=l2(self.inter_conv_l2_reg),
                activity_regularizer=None,
                trainable=True,
                name='inter-conv' + str(i),
                reuse=None)
        # deconv layers
        for i in range(self.nb_deconv_layers):
            activation = None if (i == self.nb_deconv_layers - 1) \
                else tf.nn.tanh
            kernel_num = 1 if (i == self.nb_deconv_layers - 1) \
                else self.nb_deconv_kernels
            h = tf.layers.conv2d_transpose(
                inputs=h,
                filters=kernel_num,
                kernel_size=(1, self.deconv_kernel_size),
                strides=(1, self.subsample_rate),
                padding='same',
                data_format='channels_last',
                activation=activation,
                use_bias=True,
                kernel_initializer=xavier_init,
                bias_initializer=zero_init,
                kernel_regularizer=l2(self.deconv_l2_reg),
                bias_regularizer=l2(self.deconv_l2_reg),
                activity_regularizer=None,
                trainable=True,
                name='deconv' + str(i),
                reuse=None)
        # cropping layer
        h = self._crop(h)
        h = tf.reshape(h, [self.batch_size, self.output_len],
                       name='output_reshape')
        self.predicted_y = h
        # calculate negative ccc
        self.loss = CCC_NEG_TENSOR.compute(self.input_y, self.predicted_y)
        self.loss = self.loss + sum(
            tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

    @staticmethod
    def get_test_metrics():
        return [RMSE, CCC]

    @staticmethod
    def get_selection_metric():
        return CCC
