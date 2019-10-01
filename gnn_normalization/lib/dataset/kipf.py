import os
import pickle as pkl
import sys

import networkx as nx
import numpy as np
import scipy.sparse as sp


# Code borrowed from
# https://github.com/tkipf/gcn/blob/master/gcn/utils.py


DATASET_PATH = os.path.join(
    os.path.dirname(__file__),
    'data',
    'kipf')


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def load_raw_dataset(dataset_name):
    objects = []
    for name in ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']:
        fname = os.path.join(DATASET_PATH,
                             "ind." + dataset_name.lower() + "." + name)
        with open(fname, 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    return objects


def load_test_index(dataset_name):
    test_idx_file_name = os.path.join(DATASET_PATH,
                                      "ind." + dataset_name + ".test.index")
    test_idx = parse_index_file(test_idx_file_name)
    return test_idx


def load(dataset_name):
    x, y, tx, ty, allx, ally, graph = load_raw_dataset(dataset_name)
    test_idx_reorder = load_test_index(dataset_name)

    # test_idx_reorder contains indices of test samples.
    # test_idx_reorder is not sorted.
    # tx and ty assume the order of test_idx_reorder
    # (i.e., tx[0] and ty[0] are the feature vector and the label
    # for a sample with index test_idx_reorder[0]).
    # We reorder the test dataset so that test samples are
    # sorted according to their indices.
    # It is expected that indices of test samples are contiguous.
    # However, some dataset (specifically, citeseer) have
    # missing indices. Therefore, we add zeros for missing test samples.

    test_idx_range = np.sort(test_idx_reorder)
    test_idx_range_full = range(min(test_idx_reorder),
                                max(test_idx_reorder) + 1)
    tx_extended = sp.lil_matrix((len(test_idx_range_full), allx.shape[1]))
    tx_extended[test_idx_range - min(test_idx_range), :] = tx
    tx = tx_extended
    ty_extended = np.zeros((len(test_idx_range_full), ally.shape[1]))
    ty_extended[test_idx_range - min(test_idx_range), :] = ty
    ty = ty_extended
    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    idx_test = np.array(test_idx_range.tolist())
    idx_train = np.array(range(len(y)))
    idx_val = np.array(range(len(y), len(y) + 500))
    indices = idx_train, idx_val, idx_test
    return features, labels, adj, indices
