from callbacks import Callback


class MetricLogger(Callback):
    """Class for saving and printing metrics."""
    def __init__(self, file_path, metrics, dataset='tr'):

        super(MetricLogger, self).__init__()

        if dataset != 'tr' and dataset != 'dev' and dataset != 'te':
            raise Exception(
                'MetricLogger: dataset must be \'tr\' or \'dev\' or \'te\'')

        self.file_path = file_path
        self.metrics = metrics
        self.dataset = dataset
        self.file = None

    def _on_train_begin(self):
        self.file = open(self.file_path, 'w')

    def _on_epoch_end(self):
        self.compute_metrics(self.metrics, self.dataset)
        self.file.write('epoch = ' + str(self.epoch) + '\n')
        print('------------')
        print('epoch = ' + str(self.epoch))
        print(self.dataset + ' metrics')
        self.save_metrics(self.file, self.metrics, self.dataset)
        self.print_metrics(self.metrics, self.dataset)
        print('------------')

    def _on_train_end(self):
        self.file.close()
