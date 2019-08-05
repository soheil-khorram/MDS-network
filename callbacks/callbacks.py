# Author: Soheil Khorram
# License: Simplified BSD

"""
Callback functions adapted from Keras deep learning library.
"""


class CallbackList(object):
    """
    Container abstracting a list of callbacks.
    """

    def __init__(self, callbacks=[]):
        self.callbacks = [c for c in callbacks]

    def append(self, callback):
        self.callbacks.append(callback)

    def set_model(self, model):
        for callback in self.callbacks:
            callback.set_model(model)

    def on_train_begin(self):
        for callback in self.callbacks:
            callback.on_train_begin()

    def on_epoch_begin(self, epoch):
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch)

    def on_batch_begin(self, batch):
        for callback in self.callbacks:
            callback.on_batch_begin(batch)

    def on_batch_end(self, batch_loss):
        for callback in self.callbacks:
            callback.on_batch_end(batch_loss)

    def on_epoch_end(self, preds_tr, preds_dev, preds_te):
        for callback in self.callbacks:
            callback.on_epoch_end(preds_tr, preds_dev, preds_te)

    def on_train_end(self):
        for callback in reversed(self.callbacks):
            callback.on_train_end()

    def __iter__(self):
        return iter(self.callbacks)


class Callback(object):
    """
    Abstract base class used to build new callbacks.
    """

    def __init__(self):
        self.model = None
        self.epoch = -1
        self.batch = None
        self.batch_loss = None
        self.preds_tr = None
        self.preds_dev = None
        self.preds_te = None

    def set_model(self, model):
        self.model = model

    def on_train_begin(self):
        self._on_train_begin()

    def on_epoch_begin(self, epoch):
        self.epoch = epoch
        self._on_epoch_begin()

    def on_batch_begin(self, batch):
        self.batch = batch
        self._on_batch_begin()

    def on_batch_end(self, batch_loss):
        self.batch_loss = batch_loss
        self._on_batch_end()

    def on_epoch_end(self, preds_tr, preds_dev, preds_te):
        self.preds_tr = preds_tr
        self.preds_dev = preds_dev
        self.preds_te = preds_te
        self._on_epoch_end()

    def on_train_end(self):
        self._on_train_end()

    def get_predictions(self, dataset):
        preds = None
        if dataset == 'tr':
            preds = self.preds_tr
        if dataset == 'dev':
            preds = self.preds_dev
        if dataset == 'te':
            preds = self.preds_te
        return preds

    def compute_metric(self, metric, dataset):
        preds = self.get_predictions(dataset)
        metric_value = preds.compute_metric(metric)
        return metric_value

    def compute_metrics(self, metrics, dataset):
        preds = self.get_predictions(dataset)
        metric_values = preds.compute_metrics(metrics)
        return metric_values

    def save_metrics(self, file, metrics, dataset):
        preds = self.get_predictions(dataset)
        preds.save_metrics(file, metrics)

    def print_metrics(self, metrics, dataset):
        preds = self.get_predictions(dataset)
        preds.print_metrics(metrics)

    def _on_train_begin(self):
        pass

    def _on_epoch_begin(self):
        pass

    def _on_batch_begin(self):
        pass

    def _on_batch_end(self):
        pass

    def _on_epoch_end(self):
        pass

    def _on_train_end(self):
        pass
