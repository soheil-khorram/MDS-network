from callbacks import Callback
import numpy as np
import os


class PredictionsSaver(Callback):
    def __init__(self, out_dir_path, dataset='dev'):
        super(PredictionsSaver, self).__init__()
        if dataset != 'tr' and dataset != 'dev' and dataset != 'te':
            raise Exception(
                'PredictionsSaver: dataset must be \'tr\' or \'dev\' or \'te\'')
        self.out_dir_path = out_dir_path
        self.dataset = dataset

    def _on_train_begin(self):
        if not os.path.exists(self.out_dir_path):
            os.makedirs(self.out_dir_path, 0o777)

    def _on_epoch_end(self):
        file_path = self.out_dir_path + '/' + str(self.epoch) + '.bin'
        preds = self.get_predictions(self.dataset)
        preds.save(file_path)
