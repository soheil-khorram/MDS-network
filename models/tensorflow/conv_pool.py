"""Class for a convolution neural network with pooling."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from models.model import TensorflowModel
import tensorflow as tf

l2 = tf.contrib.layers.l2_regularizer


class ConvPool(TensorflowModel):
    """Class for a convolution neural network with pooling."""

    def __init__(self, opts):
        """Construct a ConvPool model."""
        super(ConvPool, self).__init__(opts)
        self.input_dim = opts.input_dim
        self.output_dim = opts.output_dim
        self.nb_conv_layers = opts.nb_conv_layers
        self.nb_conv_kernels = opts.nb_conv_kernels
        self.conv_kernel_size = opts.conv_kernel_size
        self.conv_l2_reg = opts.conv_l2_reg
        self.nb_dense_layers = opts.nb_dense_layers
        self.dense_layer_width = opts.dense_layer_width
        self.dense_l2_reg = opts.dense_l2_reg

    def construct(self):
        """Construct a ConvPool layer."""
        self.input_x = tf.placeholder(tf.float32, [None, None, self.input_dim],
                                      name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, self.output_dim],
                                      name="input_y")
        xavier_init = tf.contrib.layers.xavier_initializer()
        zero_init = tf.zeros_initializer()
        h = self.input_x
        # conv layers
        for i in range(self.nb_conv_layers):
            h = tf.layers.conv1d(inputs=h,
                                 filters=self.nb_conv_kernels,
                                 kernel_size=self.conv_kernel_size,
                                 padding='same',
                                 data_format='channels_last',
                                 dilation_rate=1,
                                 activation=tf.nn.tanh,
                                 use_bias=True,
                                 kernel_initializer=xavier_init,
                                 bias_initializer=zero_init,
                                 kernel_regularizer=l2(self.conv_l2_reg),
                                 bias_regularizer=l2(self.conv_l2_reg),
                                 trainable=True,
                                 name='conv' + str(i))
        # max pool layer
        h = tf.reduce_max(h, axis=1, name='max_pool')
        # dense layers
        for i in range(self.nb_dense_layers):
            h = tf.layers.dense(inputs=h,
                                units=self.dense_layer_width,
                                kernel_initializer=xavier_init,
                                bias_initializer=zero_init,
                                kernel_regularizer=l2(self.dense_l2_reg),
                                bias_regularizer=l2(self.dense_l2_reg),
                                activation=tf.nn.tanh,
                                name='dense' + str(i))
        # linear layer
        h = tf.layers.dense(inputs=h,
                            units=self.output_dim,
                            kernel_initializer=xavier_init,
                            bias_initializer=zero_init,
                            kernel_regularizer=l2(self.dense_l2_reg),
                            bias_regularizer=l2(self.dense_l2_reg),
                            activation=None,
                            name='dense_logit')
        logits = h
        self.predicted_y = tf.nn.softmax(logits, name="predicted_y")
        # calculate cross-entropy loss
        with tf.name_scope("cross_entropy"):
            losses = tf.nn.softmax_cross_entropy_with_logits(
                logits=logits, labels=self.input_y)
            self.loss = tf.reduce_mean(losses)
