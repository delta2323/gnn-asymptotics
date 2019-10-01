import numpy as np
import pytest
import scipy.sparse as sp

from lib.dataset import noise


def make_adj(N):
    adj = np.triu(np.random.randint(2, size=(N, N)), 1)
    adj = adj + adj.T
    return adj


GRAPH_INFO = {
    'citeseer': {
        'node_num': 3327,
        'edge_num': 9982,
        'self_loop_num': 2
    },
    'cora': {
        'node_num': 2708,
        'edge_num': 4990,
        'self_loop_num': 0
    }
}


@pytest.fixture
def adj_citeseer():
    return make_adj(GRAPH_INFO['citeseer']['node_num'])


def test_add_noise_citeseer(adj_citeseer):
    actual = noise.add_noise(adj_citeseer, 'citeseer')
    diff = actual - adj_citeseer

    np.testing.assert_array_equal(diff, diff.T)
    assert diff.sum() == GRAPH_INFO['citeseer']['edge_num']
    assert np.diag(diff).sum() == GRAPH_INFO['citeseer']['self_loop_num']


@pytest.fixture
def adj_citeseer_sparse():
    adj = make_adj(GRAPH_INFO['citeseer']['node_num'])
    return sp.csr_matrix(adj)


def test_add_noise_citeseer_sparse(adj_citeseer_sparse):
    actual = noise.add_noise(adj_citeseer_sparse, 'citeseer')
    assert type(actual) is sp.csr_matrix

    diff = actual - adj_citeseer_sparse
    assert (diff != diff.T).nnz == 0
    assert diff.sum() == GRAPH_INFO['citeseer']['edge_num']
    diff_diag = sp.diags(diff.diagonal())
    assert diff_diag.sum() == GRAPH_INFO['citeseer']['self_loop_num']


@pytest.fixture
def adj_cora():
    return make_adj(GRAPH_INFO['cora']['node_num'])


def test_add_noise_cora(adj_cora):
    actual = noise.add_noise(adj_cora, 'cora')
    diff = actual - adj_cora

    np.testing.assert_array_equal(diff, diff.T)
    assert diff.sum() == GRAPH_INFO['cora']['edge_num']
    assert np.diag(diff).sum() == GRAPH_INFO['cora']['self_loop_num']


@pytest.fixture
def adj_cora_sparse():
    adj = make_adj(GRAPH_INFO['cora']['node_num'])
    return sp.csr_matrix(adj)


def test_add_noise_cora_sparse(adj_cora_sparse):
    actual = noise.add_noise(adj_cora_sparse, 'cora')
    assert type(actual) is sp.csr_matrix

    diff = actual - adj_cora_sparse
    assert (diff != diff.T).nnz == 0
    assert diff.sum() == GRAPH_INFO['cora']['edge_num']
    diff_diag = sp.diags(diff.diagonal())
    assert diff_diag.sum() == GRAPH_INFO['cora']['self_loop_num']
