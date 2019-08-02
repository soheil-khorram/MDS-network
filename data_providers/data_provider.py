"""Base class that provides data."""
import numpy as np


class Batch(object):
    """
    An structure contains a list of strings as utt_ids,
    a numpy matrix x and a numpy matrix y.
    """

    def __init__(self):
        """Construct a batch."""
        self.utts = []
        self.x = []
        self.y = []

    def append(self, item):
        this_utt, this_x, this_y = item
        self.utts.append(this_utt)
        self.x.append(this_x)
        self.y.append(this_y)

    def finalize(self):
        self.x = np.array(self.x)
        self.y = np.array(self.y)


class DataProvider(object):
    """Base class that provides data."""

    @staticmethod
    def parse_options(parser):
        """Parse options for DataProvider."""
        parser.add_argument('--batch-size', type=int, default=1)

    def __init__(self, opts, dataset):
        """Init data provider."""
        self.nb_samples = None
        self.batch_size = opts.batch_size
        if dataset == 'tr':
            self.dataset = 'tr'
            self.shuffle = True
        elif dataset == 'dev':
            self.dataset = 'dev'
            self.shuffle = False
        elif dataset == 'te':
            self.dataset = 'te'
            self.shuffle = False
        else:
            raise Exception(
                'DataProvider: dataset must be \'tr\', \'dev\' or \'te\'')

    def load(self, opts):
        """Load the data."""
        if self.dataset == 'tr':
            self.nb_samples = self.load_tr(opts)
        elif self.dataset == 'dev':
            self.nb_samples = self.load_dev(opts)
        elif self.dataset == 'te':
            self.nb_samples = self.load_te(opts)
        else:
            raise Exception(
                'DataProvider.load: dataset must be \'tr\', \'dev\' or \'te\'')
        self.__iter__()
        pass

    def __iter__(self):
        """Iterator initializer."""
        if self.nb_samples is None:
            raise Exception(
                'DataProvider.__iter__: please load the dataprovider first')
        self.ind = 0
        if self.shuffle:
            self.shuffler = np.random.permutation(self.nb_samples)
        else:
            self.shuffler = np.array(range(self.nb_samples))
        return self

    def next(self):
        """Return the next item in iteator."""
        if self.nb_samples is None:
            raise Exception(
                'DataProvider.__iter__: please load the dataprovider first')
        if self.ind >= self.nb_samples:
            raise StopIteration
        start_ind = self.ind
        end_ind = min(start_ind + self.batch_size, self.nb_samples)
        self.ind = end_ind
        batch = Batch()
        for i in range(start_ind, end_ind):
            batch.append(self.get_sample(self.shuffler[i]))
        batch.finalize()
        return batch

    def load_tr(self, opts):
        """Load the train data and return the number of samples."""
        pass

    def load_dev(self, opts):
        """Load the dev data and return the number of samples."""
        pass

    def load_te(self, opts):
        """Load the test data and return the number of samples."""
        pass

    def get_sample(self, i):
        """Return i-th sample (utts, x, y)."""
        pass
