from callbacks import Callback
import numpy as np


class BestResultSaver(Callback):
    def __init__(self, out_file_path, metric, dataset='dev', mode='max'):
        super(BestResultSaver, self).__init__()
        if dataset != 'tr' and dataset != 'dev' and dataset != 'te':
            raise Exception(
                'BestResultSaver: dataset must be \'tr\' or \'dev\' or \'te\'')
        self.out_file_path = out_file_path
        self.metric = metric
        self.dataset = dataset
        self.best_epoch = -1
        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            raise Exception('unkown mode: choose from \{min,max\}')

    def _on_epoch_end(self):
        metric_value = self.compute_metric(self.metric, self.dataset)
        if self.monitor_op(metric_value, self.best):
            print(self.dataset + ' ' + self.metric.get_name() + ' is getting better.')
            self.best = metric_value
            self.best_epoch = self.epoch
            f = open(self.out_file_path, 'w')
            f.write('Best epoch ind: ' + str(self.best_epoch) + '\n')
            f.write('Best ' + self.dataset + ' ' + self.metric.get_name() + ': ' + str(self.best))
            f.close()
        print('Best epoch ind: ' + str(self.best_epoch))
        print('Best ' + self.dataset + ' ' + self.metric.get_name() + ': ' + str(self.best))
