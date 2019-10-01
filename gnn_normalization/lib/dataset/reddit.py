import numpy as np
import scipy.sparse as sp

from lib.dataset import normalization as N
from lib.dataset import transductive_dataset as td


def loadRedditFromNPZ(dataset_dir):
    adj = sp.load_npz(dataset_dir + "reddit_adj.npz")
    data = np.load(dataset_dir + "reddit.npz")

    return (adj, data['feats'],
            (data['y_train'], data['y_val'], data['y_test']),
            (data['train_index'], data['val_index'], data['test_index']))


def load_reddit_data(data_path="data/", normalization="AugNormAdj", cuda=True):
    adj, features, y, indices = loadRedditFromNPZ(data_path)
    y_train, y_val, y_test = y
    train_index, val_index, test_index = indices

    labels = np.zeros(adj.shape[0])
    labels[train_index] = y_train
    labels[val_index] = y_val
    labels[test_index] = y_test
    adj = adj + adj.T + sp.eye(adj.shape[0])
    train_adj = adj[train_index, :][:, train_index]

    features = (features-features.mean(dim=0))/features.std(dim=0)
    adj_normalizer = N.fetch_normalization(normalization)
    adj = adj_normalizer(adj)
    train_adj = adj_normalizer(train_adj)

    return td.TransductiveDataset(features, adj, labels, indices)
