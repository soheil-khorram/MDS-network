# Author: Soheil Khorram
# License: Simplified BSD

"""A class designed for storing and using the results of model predictions."""

import pickle


class Predictions(object):

    def __init__(self, utts, y_true, y_pred):
        self._utts = utts
        self._y_true = y_true
        self._y_pred = y_pred
        self._metric_dic = dict()

    def save(self, path):
        with open(path, 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(path):
        preds = None
        with open(path, 'rb') as input:
            preds = pickle.load(input)
        return preds

    def compute_metric(self, metric):
        """Compute metric and also cashes the result."""
        if metric not in self._metric_dic:
            self._metric_dic[metric] = metric.compute(self._y_true,
                                                      self._y_pred)
        return self._metric_dic[metric]

    def compute_metrics(self, metrics):
        results = []
        for metric in metrics:
            results.append(self.compute_metric(metric))
        return results

    def save_metrics(self, outfile, metrics=None):
        if metrics is None:
            metrics = self._metric_dic.keys()
        metric_values = self.compute_metrics(metrics)
        for m, m_value in zip(metrics, metric_values):
            outfile.write('{}: {}\n'.format(m.get_name(), m_value))

    def print_metrics(self, metrics=None):
            if metrics is None:
                metrics = self._metric_dic.keys()
            metric_values = self.compute_metrics(metrics)
            for m, m_value in zip(metrics, metric_values):
                print('{}: {}'.format(m.get_name(), m_value))
