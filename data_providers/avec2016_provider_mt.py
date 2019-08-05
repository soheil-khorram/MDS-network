# Author: Soheil Khorram
# License: Simplified BSD

from data_provider import DataProvider
import numpy as np
import glob


class Avec2016ProviderMT(DataProvider):
    @staticmethod
    def parse_options(parser):
        DataProvider.parse_options(parser)
        parser.add_argument('--mfb-dir',
                            default='/z/public/data/AVEC_2016/features_mfb')
        parser.add_argument('--task', type=str, default='arousal')
        parser.add_argument(
            '--labels-dir',
            default='/z/public/data/AVEC_2016/ratings_individual_centered')

    def __init__(self, opts, dataset):
        super(Avec2016ProviderMT, self).__init__(opts, dataset)
        self.task = opts.task
        self.utts = []
        self.utts_tr = []
        self.utts_dev = []
        self.utts_te = []
        self.utt2feats = {}
        self.utt2label = {}
        self.dataset = dataset

        # get all file names
        files_lists = glob.glob(opts.mfb_dir + '/*.npy')

        # for each file, get features and labels
        for curr_file in files_lists:
            curr_file = curr_file.split('/')[-1].split('.')[0]

            if 'train' in curr_file:
                self.utts_tr.append(curr_file)
            elif 'dev' in curr_file:
                self.utts_dev.append(curr_file)
            elif 'te' in curr_file:
                self.utts_te.append(curr_file)
            else:
                err_str = 'cannot classify file into on of {train, dev, test}'
                raise ValueError(err_str)

            temp_mfb = np.load(opts.mfb_dir + '/' + curr_file + '.npy')
            try:
                temp_label = np.loadtxt(opts.labels_dir + '/' + opts.task +
                                        '/' + curr_file + '.csv',
                                        delimiter=',')
                temp_label = np.transpose(temp_label)
            except:
                temp_label = None
            temp_mfb = self._stack(temp_mfb, 4, 4)
            temp_mfb = self._normalize(temp_mfb, axis=0)
            self.utt2feats[curr_file] = temp_mfb
            self.utt2label[curr_file] = temp_label

    @staticmethod
    def _normalize(feats_in, axis=0):
        m = np.mean(feats_in, axis=axis)
        s = np.std(feats_in, axis=axis)
        return (feats_in-m) / s

    @staticmethod
    def _stack(x_in, stack_frames, skip_frames):
        i = 0
        stacked = []
        stacked_dim = x_in.shape[1] * stack_frames
        while i < x_in.shape[0]:
            temp = list(x_in[i:i+stack_frames].flatten())
            # Pad if necessary
            if len(temp) < stacked_dim:
                frames_to_pad = (stacked_dim - len(temp)) / x_in.shape[1]
                temp += list(x_in[-1]) * frames_to_pad
            stacked.append(temp)
            i += skip_frames
        stacked.append(temp)
        return np.array(stacked, dtype=x_in.dtype)

    def load_tr(self, opts):
        self.utts = self.utts_tr
        return len(self.utts)

    def load_dev(self, opts):
        self.utts = self.utts_dev
        return len(self.utts)

    def load_te(self, opts):
        self.utts = self.utts_te
        return len(self.utts)

    def get_sample(self, i):
        return (self.dataset + '_' + self.utts[i],
                self.utt2feats[self.utts[i]], self.utt2label[self.utts[i]])
