import numpy as np
import scipy.sparse as sp


def gcn_normalize(features, adj):
    adj = _augmented_normalize(adj)
    features = _row_normalize(features)
    return features, adj


def _augmented_normalize(adj):
    """Computes the augmented normalized adjacency matrix.

       For an adjacency matrix A, the augmented normalized
       adjacency matrix is defined as:
           \tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2},
       where \tilde{A}:=A+I and D is the degree matrix of
       \tilde{A}.

    Args:
        adj(2d ndarray of size (C, C)): adjacency matrix.

    Returns:
        the augmented normalized adjacency matrix of adj.
    """

    adj = adj + sp.eye(adj.shape[0])
    adj = sp.coo_matrix(adj)
    row_sum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()


def _row_normalize(mx):
    """Computes row-normalize sparse matrix.

       Divide each row by the sum of each row.
       Therefore, the result matrix is row-normalized in the sense that
       the sum of each row equals to 1.

    Args:
        mx(scipy sparse csr matrix): input matrix.

    Returns:
        row-normalized sparse matrix of the same size.
    """
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx
