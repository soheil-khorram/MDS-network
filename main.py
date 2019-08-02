"""Main python file."""

import argparse
from os import makedirs, environ
from os.path import abspath, join, isdir, isfile

if environ['DATA_PROVIDER'] == 'avec2016_provider':
    from data_providers.avec2016_provider import Avec2016Provider \
        as DataProvider
if environ['DATA_PROVIDER'] == 'avec2016_provider_mt':
    from data_providers.avec2016_provider_mt import Avec2016ProviderMT \
        as DataProvider

if environ['MODEL'] == 'conv_deconv':
    from models.tensorflow.conv_deconv import ConvDeconv as Model
if environ['MODEL'] == 'preliminary_cnn':
    from models.tensorflow.preliminary_cnn import PreliminaryCnn as Model
if environ['MODEL'] == 'delay_attention':
    from models.tensorflow.delay_attention \
        import DelayAttention as Model
if environ['MODEL'] == 'sinc_kernel':
    from models.tensorflow.sinc_kernel import SincKernel as Model
if environ['MODEL'] == 'conv_deconv_global_sigma':
    from models.tensorflow.density_nets.conv_deconv_global_sigma \
        import ConvDeconvGlobalSigma as Model
if environ['MODEL'] == 'conv_deconv_variable_sigma':
    from models.tensorflow.density_nets.conv_deconv_variable_sigma \
        import ConvDeconvVariableSigma as Model
if environ['MODEL'] == 'conv_deconv_mixture_density':
    from models.tensorflow.density_nets.conv_deconv_mixture_density \
        import ConvDeconvMixtureDensity as Model
if environ['MODEL'] == 'conv_deconv_gaussian_process':
    from models.tensorflow.density_nets.conv_deconv_gaussian_process \
        import ConvDeconvGaussianProcess as Model


def parse_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-dir')
    parser.add_argument('--exp-done-file-name', type=str, default='exp.done')
    parser.add_argument('--seed', type=int, default=0)
    DataProvider.parse_options(parser)
    Model.parse_options(parser)
    return parser.parse_args()


def prepare_directories(opts):
    opts.exp_dir = abspath(opts.exp_dir)
    if not isdir(opts.exp_dir):
        makedirs(opts.exp_dir, 0o777)
    opts.exp_done_file = join(opts.exp_dir, opts.exp_done_file_name)
    return opts


def prepare_data_providers(opts):
    data_tr = DataProvider(opts, 'tr')
    data_tr.load(opts)
    data_dev = DataProvider(opts, 'dev')
    data_dev.load(opts)
    data_te = DataProvider(opts, 'te')
    data_te.load(opts)
    return data_tr, data_dev, data_te


def main(opts):

    opts = prepare_directories(opts)
    if isfile(opts.exp_done_file):
        return
    data_tr, data_dev, data_te = prepare_data_providers(opts)
    model = Model(opts)
    model.construct()
    model.train(data_tr, data_dev, data_te)
    open(opts.exp_done_file, 'a').close()


if __name__ == '__main__':
    options = parse_options()
    main(options)
