from os import makedirs
from os.path import join, isdir
from callbacks.metric_logger import MetricLogger
from callbacks.best_result_saver import BestResultSaver
from callbacks.best_predictions_saver import BestPredictionsSaver
from callbacks.detailed_metric_logger import DetailedMetricLogger


class Model(object):
    """Base class for all models."""

    @staticmethod
    def parse_options(parser):
        """Parse options for Model"""
        parser.add_argument('--best-model-file-name', type=str,
                            default='best_model.bin')
        parser.add_argument('--log-folder-name', type=str,
                            default='log')
        parser.add_argument('--predictions-folder-name', type=str,
                            default='predictions')
        parser.add_argument('--test-preds-file-name', type=str,
                            default='test_predictions.bin')
        parser.add_argument('--test-result-file-name', type=str,
                            default='test_result.txt')

    def __init__(self, opts):
        """Define all options of the model."""
        self.log_dir = join(opts.exp_dir, opts.log_folder_name)
        if not isdir(self.log_dir):
            makedirs(self.log_dir, 0o777)
        self.predictions_dir = join(opts.exp_dir, opts.predictions_folder_name)
        if not isdir(self.predictions_dir):
            makedirs(self.predictions_dir, 0o777)
        self.best_model_path = join(opts.exp_dir, opts.best_model_file_name)
        self.test_preds_file_path = \
            join(opts.exp_dir, opts.test_preds_file_name)
        self.test_result_file_path = \
            join(opts.exp_dir, opts.test_result_file_name)

    def test(self, data_te):
        """predict model using data_te input, save predictions and metrics."""
        predictions = self.predict(data_te)
        test_metrics = self.get_test_metrics()
        predictions.compute_metrics(test_metrics)
        predictions.save(self.test_preds_file_path)
        predictions.save_metrics(self.test_result_file_path)

    def construct(self):
        """Construct a model."""
        pass

    def train_on_batch(self, x_batch, y_batch):
        """Update parameters using one batch of data."""
        pass

    def train(self, data_tr, data_dev, data_te):
        """Train the model."""
        pass

    def predict_on_batch(self, x_batch):
        """Predict the network output for one batch of data."""
        pass

    def predict(self, data):
        """Predict the network output for all samples in data."""
        pass

    def define_callbacks(self):
        metrics = self.get_test_metrics()
        callbacks = []
        callbacks.append(MetricLogger(self.log_dir + '/tr_log.txt',
                                      metrics, dataset='tr'))
        callbacks.append(MetricLogger(self.log_dir + '/te_log.txt',
                                      metrics, dataset='te'))
        callbacks.append(MetricLogger(self.log_dir + '/dev_log.txt',
                                      metrics, dataset='dev'))
        # callbacks.append(PredictionsSaver(
        #     self.predictions_dir + '/tr', 'tr'))
        # callbacks.append(PredictionsSaver(
        #     self.predictions_dir + '/te', 'te'))
        # callbacks.append(PredictionsSaver(
        #     self.predictions_dir + '/dev', 'dev'))
        callbacks.append(BestPredictionsSaver(
            self.predictions_dir + '/best_test_preds.bin',
            self.get_selection_metric(), dataset_to_be_optimized='dev',
            mode_to_be_optimized='max', dataset_to_be_predicted='te'))
        callbacks.append(
            BestResultSaver(self.log_dir + '/best_dev_result.txt',
                            self.get_selection_metric(),
                            dataset='dev', mode='max')
        )
        callbacks.append(
            DetailedMetricLogger(
                self.log_dir + '/detailed_dev_log.txt', metrics, dataset='dev')
        )
        return callbacks

    @staticmethod
    def get_test_metrics():
        """Return a list containing all test metrics."""
        pass

    @staticmethod
    def get_selection_metric():
        """Return a metric for selecting best model."""
        pass
