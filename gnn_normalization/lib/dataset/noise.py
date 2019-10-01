import os

import scipy.sparse as sp

DIR_NAME = os.path.join(
    os.path.abspath(os.path.dirname(__file__)),
    'data',
    'noisy_diff')


NOISY_DIFF_PATH = {
    'citeseer': {
        '2500': os.path.join(
            DIR_NAME, 'citeseer-diff-2500.npz'),
        '5000': os.path.join(
            DIR_NAME, 'citeseer-diff-5000.npz'),
        '10000': os.path.join(
            DIR_NAME, 'citeseer-diff-10000.npz')},
    'cora': {
        '2500': os.path.join(
            DIR_NAME, 'cora-diff-2500.npz'),
        '5000': os.path.join(
            DIR_NAME, 'cora-diff-5000.npz'),
        '10000': os.path.join(
            DIR_NAME, 'cora-diff-10000.npz')},
    'pubmed': {
        '10000': os.path.join(
            DIR_NAME, 'pubmed-diff-10000.npz'),
        '25000': os.path.join(
            DIR_NAME, 'pubmed-diff-25000.npz')},
    }


def _load_noisy_diff(dataset_name):
    if dataset_name == 'citeseer':
        fname = NOISY_DIFF_PATH['citeseer']['5000']
    elif dataset_name == 'cora':
        fname = NOISY_DIFF_PATH['cora']['2500']
    elif dataset_name == 'pubmed':
        fname = NOISY_DIFF_PATH['pubmed']['25000']
    else:
        raise ValueError('invalid dataset name: {}'.format(dataset_name))
    diff = sp.load_npz(fname)
    return diff


def add_noise(adj, dataset_name):
    diff = _load_noisy_diff(dataset_name)
    return adj + diff
