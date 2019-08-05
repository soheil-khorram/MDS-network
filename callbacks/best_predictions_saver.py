# Author: Soheil Khorram
# License: Simplified BSD

from callbacks import Callback
import numpy as np


class BestPredictionsSaver(Callback):
    def __init__(self, out_file_path, metric_to_be_optimized,
                 dataset_to_be_optimized='dev', mode_to_be_optimized='max',
                 dataset_to_be_predicted='te'):
        super(BestPredictionsSaver, self).__init__()
        if dataset_to_be_optimized != 'tr' and dataset_to_be_optimized != 'dev' and dataset_to_be_optimized != 'te':
            raise Exception(
                'BestPredictionsSaver: dataset_to_be_optimized must be \'tr\' or \'dev\' or \'te\'')
        if dataset_to_be_predicted != 'tr' and dataset_to_be_predicted != 'dev' and dataset_to_be_predicted != 'te':
            raise Exception(
                'BestPredictionsSaver: dataset_to_be_predicted must be \'tr\' or \'dev\' or \'te\'')
        self.out_file_path = out_file_path
        self.metric = metric_to_be_optimized
        self.dataset_to_be_optimized = dataset_to_be_optimized
        if mode_to_be_optimized == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode_to_be_optimized == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            raise Exception('unkown mode: choose from \{min,max\}')
        self.dataset_to_be_predicted = dataset_to_be_predicted

    def _on_epoch_end(self):
        metric_value = self.compute_metric(self.metric, self.dataset_to_be_optimized)
        if self.monitor_op(metric_value, self.best):
            print('Saving ' + self.dataset_to_be_predicted + ' preds')
            self.best = metric_value
            preds = self.get_predictions(self.dataset_to_be_predicted)
            preds.save(self.out_file_path)
