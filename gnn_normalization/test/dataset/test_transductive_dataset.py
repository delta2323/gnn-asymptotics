import numpy as np
import pytest

from lib.dataset import transductive_dataset as td


def make_adj(N):
    ret = np.triu(np.random.randint(1, size=(N, N)), 1).astype(np.int32)
    return ret + ret.T


def make_partition(N):
    indices = np.random.permutation(range(N))
    train_idx = indices[:int(N * 0.8)]
    val_idx = indices[int(N * 0.8):int(N * 0.9)]
    test_idx = indices[int(N * 0.9):]
    return train_idx, val_idx, test_idx


def make_raw_dataset(N, C, K):
    X = np.random.standard_normal((N, C)).astype(np.float32)
    A = make_adj(N)

    y = np.zeros((N, K), dtype=np.int32)
    y_int = np.random.randint(K)
    y[y_int] = 1

    indices = make_partition(N)
    return X, A, y, indices


@pytest.fixture
def raw_dataset():
    N = 10
    C = 3
    K = 7
    return make_raw_dataset(N, C, K)


def test_td(raw_dataset):
    dataset = td.TransductiveDataset(*raw_dataset)
    assert len(dataset) == 10
