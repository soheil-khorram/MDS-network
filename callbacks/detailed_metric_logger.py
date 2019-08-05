# Author: Soheil Khorram
# License: Simplified BSD

from callbacks import Callback
import numpy as np
import os


class DetailedMetricLogger(Callback):
    def __init__(self, file_path, metrics, dataset='dev'):
        super(DetailedMetricLogger, self).__init__()
        if dataset != 'tr' and dataset != 'dev' and dataset != 'te':
            raise Exception(
                'DetailedMetricLogger: dataset must be \'tr\' or \'dev\' or \'te\'')
        self.file_path = file_path
        self.metrics = metrics
        self.dataset = dataset
        self.file = None

    def _on_train_begin(self):
        self.file = open(self.file_path, 'w')

    def _on_epoch_end(self):
        self.file.write('epoch = ' + str(self.epoch) + '\n')
        preds = self.get_predictions(self.dataset)

        self.file.write('utts:\t')
        for utt in preds._utts:
            self.file.write(utt + '\t')
        self.file.write('\n')

        for metric in self.metrics:
            self.file.write(metric.get_name() + ':\t')
            for i in range(len(preds._utts)):
                metric_value = metric.compute(preds._y_true[i, :], preds._y_pred[i, :])
                self.file.write(str(metric_value) + '\t')
            self.file.write('\n')
        self.file.write('\n')
        self.file.flush()

    def _on_train_end(self):
        self.file.close()
