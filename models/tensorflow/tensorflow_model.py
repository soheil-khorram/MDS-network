"""Base class for tensorflow models."""
import tensorflow as tf
import numpy as np
from callbacks.callbacks import CallbackList
from models.predictions import Predictions
from models.model import Model


class TensorflowModel(Model):
    @staticmethod
    def parse_options(parser):
        """Parse options for TensorflowModel."""
        Model.parse_options(parser)

    """Base class for tensorflow models."""
    def __init__(self, opts):
        super(TensorflowModel, self).__init__(opts)
        self.stop_training = False
        self.input_x = None
        self.input_y = None
        self.input_lr = None
        self.predicted_y = None
        self.loss = 0
        self.sess = None
        self.train_op = None

    def _init_train_op(self):
        self.input_lr = tf.placeholder(tf.float32, shape=[])
        optimizer = tf.train.AdamOptimizer(self.input_lr)
        grads_and_vars = optimizer.compute_gradients(self.loss)
        capped_gvs =\
            [(tf.clip_by_value(grad, self.min_grad, self.max_grad), var)
             for grad, var in grads_and_vars]
        capped_gvs = grads_and_vars
        self.train_op = optimizer.apply_gradients(capped_gvs)

    def _init_train_sess(self):
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def _init_train_callbacks(self, callbacks):
        callback_list = CallbackList(callbacks)
        callback_list.set_model(self)
        callback_list.on_train_begin()
        return callback_list

    def train_on_batch(self, x_batch, y_batch):
        """Update parameters using one batch of data."""
        feed_dict = {
            self.input_x: x_batch,
            self.input_y: y_batch,
            self.input_lr: self.lr
        }
        _, loss = self.sess.run([self.train_op, self.loss], feed_dict)
        return loss

    def train(self, data_tr, data_dev, data_te):
        """Train the model."""
        self._init_train_op()
        self._init_train_sess()
        callbacks = self.define_callbacks()
        callback_list = self._init_train_callbacks(callbacks)
        callback_list.on_train_begin()

        for curr_epoch in range(self.nb_epochs):

            callback_list.on_epoch_begin(curr_epoch)
            if self.stop_training:
                break
            for batch in data_tr:
                callback_list.on_batch_begin(batch)
                batch_loss = self.train_on_batch(batch.x, batch.y)
                callback_list.on_batch_end(batch_loss)

            preds_tr = self.predict(data_tr)
            preds_dev = self.predict(data_dev)
            preds_te = self.predict(data_te)
            # run end_of_epoch functions
            callback_list.on_epoch_end(preds_tr, preds_dev, preds_te)

        callback_list.on_train_end()

    def predict_on_batch(self, x_batch):
        """Predict the network output for one batch of data."""
        feed_dict = {self.input_x: x_batch}
        predicted_y = self.sess.run(self.predicted_y, feed_dict)
        return predicted_y

    def predict(self, data):
        """Predict output labels for the input data."""
        y_pred = None
        y_true = None
        utts = []
        for batch in data:
            y_pred_batch = self.predict_on_batch(batch.x)
            if y_pred is None:
                y_pred = y_pred_batch
                y_true = batch.y
            else:
                y_pred = np.concatenate((y_pred, y_pred_batch))
                y_true = np.concatenate((y_true, batch.y))

            utts = utts + batch.utts
        return Predictions(utts, y_true, y_pred)

    def construct(self):
        """Construct a model."""
        pass

    @staticmethod
    def get_test_metrics():
        """Return a list containing all test metrics."""
        pass

    @staticmethod
    def get_selection_metric():
        """Return a metric for selecting best model."""
        pass
