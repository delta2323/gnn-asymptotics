import numpy as np

from lib.dataset import kipf
from lib.dataset import noise
import lib.dataset.normalization as N
import lib.dataset.transductive_dataset as td


def to_chch(adj, features):
    """Converts dataset to adapt Chainer Chemistry"""
    # to dense matrix
    features = features.todense().astype(np.float32)
    adj = adj.todense().astype(np.float32)

    # should add the new axis
    features = features[np.newaxis, :, :]
    adj = adj[np.newaxis, :, :]

    return adj, features


def load(dataset_name,
         gcn_normalize=False,
         noisy=False):
    """Loads citation dataset

    Args:
        dataset_name (str): choice of dataset
        gcn_normalize (Boolean): Set True if used for RSGCN

    Returns:
        TransductiveDataset: wraps all node features,
                             adjacenty matrix,
                             index for train/val/test samples,
                             and labels in one instance
    """

    features, labels, adj, indices = kipf.load(dataset_name)

    if noisy:
        adj = noise.add_noise(adj, dataset_name)

    if gcn_normalize:
        features, adj = N.gcn_normalize(features, adj)

    features, adj = to_chch(features, adj)

    return td.TransductiveDataset(
        features, adj, labels, indices)
