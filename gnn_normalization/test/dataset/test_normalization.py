import numpy as np
import pytest
import scipy.sparse as sp

from lib.dataset import normalization as N


def f(adj):
    C = len(adj)
    aug_adj = adj + np.eye(C)
    d_inv_sqrt = 1. / np.sqrt(aug_adj.sum(axis=1))
    return d_inv_sqrt[:, None] * (aug_adj) * d_inv_sqrt


@pytest.fixture
def adj_simple():
    return np.array([[1., 1., 0.],
                     [1., 1., 0.],
                     [0., 0., 1.]], dtype=np.float32)


@pytest.fixture
def adj_disconnected():
    return np.array([[1., 0., 0.],
                     [0., 0., 0.],
                     [0., 0., 1.]], dtype=np.float32)


def test_augmented_normalize_1(adj_simple):
    expect = f(adj_simple)
    actual = N._augmented_normalize(adj_simple)
    np.testing.assert_array_almost_equal(
        expect, actual.toarray())


def test_augmented_normalize_2():
    adj = np.eye(10)
    actual = N._augmented_normalize(adj)
    expect = adj
    np.testing.assert_array_almost_equal(
        expect, actual.toarray())


def test_augmented_normalize_zero_in_diagonal(
        adj_disconnected):
    actual = N._augmented_normalize(adj_disconnected)
    expect = f(adj_disconnected)
    np.testing.assert_array_almost_equal(
        expect, actual.toarray())


@pytest.fixture
def x():
    return sp.csr_matrix([[1., 2., 3.],
                          [4., 5., 6.],
                          [7., 8., 9.]],
                         dtype=np.float32)


@pytest.fixture
def diagonal():
    return sp.csr_matrix([[1., 0., 0.],
                          [0., 5., 0.],
                          [0., 0., 9.]],
                         dtype=np.float32)


@pytest.fixture
def zero_row():
    return sp.csr_matrix([[1., 0., 0.],
                          [0., 0., 0.],
                          [0., 0., 9.]],
                         dtype=np.float32)


def test_row_normalize(x):
    actual = N._row_normalize(x)
    expect = x / x.sum(axis=1)
    np.testing.assert_array_almost_equal(
        expect, actual.toarray())


def test_row_normalize_zero_row(zero_row):
    actual = N._row_normalize(zero_row)
    expect = np.array([[1., 0., 0.],
                       [0., 0., 0.],
                       [0., 0., 1.]],
                      dtype=actual.dtype)
    np.testing.assert_array_almost_equal(
        expect, actual.toarray())
