from data_provider import DataProvider
import numpy as np


class Avec2016Provider(DataProvider):
    @staticmethod
    def parse_options(parser):
        DataProvider.parse_options(parser)
        parser.add_argument('--task', type=str, default='arousal')
        parser.add_argument('--avec2016-dir', type=str,
                            default='/z/Soheil/data/constraint-kernel-cnn')

    def __init__(self, opts, dataset):
        super(Avec2016Provider, self).__init__(opts, dataset)
        self.task = opts.task
        self.avec2016_dir = opts.avec2016_dir
        self.tr_mfb_path = self.avec2016_dir + '/train/mfb.npy'
        self.dev_mfb_path = self.avec2016_dir + '/dev/mfb.npy'
        self.te_mfb_path = self.avec2016_dir + '/test/mfb.npy'
        self.tr_lab_path = self.avec2016_dir + '/train/' + self.task \
            + '/labels.npy'
        self.dev_lab_path = self.avec2016_dir + '/dev/' + self.task \
            + '/labels.npy'
        self.te_lab_path = self.avec2016_dir + '/test/' + self.task \
            + '/labels.npy'
        self.dataset = dataset

    @staticmethod
    def _normalize(x):
        r = np.full(x.shape, np.nan)
        sample_num = x.shape[0]
        for si in range(sample_num):
            this_x = x[si]
            m = this_x.mean(0)
            s = this_x.std(0)
            s[s < 0.000001] = 0.000001
            r[si] = (this_x - m) / s
        return r

    def _load_mfb(self, mfb_path):
        self.x = np.load(mfb_path)  # shape = 9*7501*160
        self.x = Avec2016Provider._normalize(self.x)
        return self.x.shape[0]

    def _load_lab(self, lab_path):
        self.y = np.load(lab_path)  # shape = 9*7501

    def load_tr(self, opts):
        self._load_lab(self.tr_lab_path)
        return self._load_mfb(self.tr_mfb_path)

    def load_dev(self, opts):
        self._load_lab(self.dev_lab_path)
        return self._load_mfb(self.dev_mfb_path)

    def load_te(self, opts):
        self._load_lab(self.te_lab_path)
        return self._load_mfb(self.te_mfb_path)

    def get_sample(self, i):
        return(self.dataset + '_utt' + str(i), self.x[i], self.y[i])
