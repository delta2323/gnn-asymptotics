import numpy as np


def make_p(lambda_):
    """Makes random symmetric matrix from eigen values."""
    N = len(lambda_)
    lambda_ = np.diag(lambda_)
    Q = _sample_orthogonal_matrix(N)
    P = np.matmul(np.matmul(Q, lambda_), Q.T)
    return P


def make_w(s):
    """Makes random matrix from singular values."""
    C = len(s)
    U = _sample_orthogonal_matrix(C)
    V = _sample_orthogonal_matrix(C)
    s = np.diag(s)
    W = np.matmul(np.matmul(U, s), V)
    return W


def _sample_orthogonal_matrix(N):
    M = np.random.uniform(-1, 1, (N, N))
    Q, _ = np.linalg.qr(M, 'complete')
    return Q
