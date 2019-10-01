import chainer
from chainer import functions
from chainer import reporter
from chainer import Variable
import chainer_chemistry
from chainer_chemistry.config import MAX_ATOMIC_NUM
from chainer_chemistry.links import GraphLinear
from chainer_chemistry.links.readout import general_readout
from chainer_chemistry.links.update import rsgcn_update
import networkx as nx
import numpy as np


class RSGCN(chainer.Chain):

    """Renormalized Spectral Graph Convolutional Network (RSGCN)

    See: Thomas N. Kipf and Max Welling, \
        Semi-Supervised Classification with Graph Convolutional Networks. \
        September 2016. \
        `arXiv:1609.02907 <https://arxiv.org/abs/1609.02907>`_

    The name of this model "Renormalized Spectral Graph Convolutional Network
    (RSGCN)" is named by us rather than the authors of the paper above.
    The authors call this model just "Graph Convolution Network (GCN)", but
    we think that "GCN" is bit too general and may cause namespace issue.
    That is why we did not name this model as GCN.

    Args:
        out_dim (int): dimension of output feature vector
        hidden_dim (int): dimension of feature vector
            associated to each atom
        n_atom_types (int): number of types of atoms
        n_layers (int): number of layers
        use_batch_norm (bool): If True, batch normalization is applied after
            graph convolution.
        readout (Callable): readout function. If None,
            `GeneralReadout(mode='sum)` is used.
            To the best of our knowledge, the paper of RSGCN model does
            not give any suggestion on readout.
        dropout_ratio (float): ratio used in dropout function.
            If 0 or negative value is set, dropout function is skipped.
        no_readout (bool): if set to True, the final embedding
            vectors are emited
    """

    def __init__(self, out_dim, hidden_dim=32, n_layers=4,
                 n_atom_types=MAX_ATOMIC_NUM,
                 use_batch_norm=False, readout=None,
                 dropout_ratio=0.5, no_readout=False,
                 initial_weight_scale=None,
                 unchain=False):
        super(RSGCN, self).__init__()
        in_dims = [hidden_dim for _ in range(n_layers)]
        out_dims = [hidden_dim for _ in range(n_layers)]
        out_dims[n_layers - 1] = out_dim
        if readout is None:
            readout = general_readout.GeneralReadout()
        with self.init_scope():
            self.embed = chainer_chemistry.links.EmbedAtomID(
                in_size=n_atom_types, out_size=hidden_dim)
            self.compress = GraphLinear(in_size=None, out_size=hidden_dim)
            self.gconvs = chainer.ChainList(
                *[rsgcn_update.RSGCNUpdate(in_dims[i], out_dims[i])
                  for i in range(n_layers)])
            if use_batch_norm:
                self.bnorms = chainer.ChainList(
                    *[chainer_chemistry.links.GraphBatchNormalization(
                        out_dims[i]) for i in range(n_layers)])
            else:
                self.bnorms = [None for _ in range(n_layers)]
            if isinstance(readout, chainer.Link):
                self.readout = readout
        if not isinstance(readout, chainer.Link):
            self.readout = readout
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dropout_ratio = dropout_ratio
        self.no_readout = no_readout

        if initial_weight_scale is not None:
            self.normalize(initial_weight_scale)
        self.unchain = unchain

        self.G = None

    def __call__(self, graph, adj, return_all_activations=False):
        """Forward propagation

        Args:
            graph (numpy.ndarray): minibatch of molecular which is
                represented with atom IDs (representing C, O, S, ...)
                `atom_array[mol_index, atom_index]` represents `mol_index`-th
                molecule's `atom_index`-th atomic number
            adj (numpy.ndarray): minibatch of adjancency matrix
                `adj[mol_index]` represents `mol_index`-th molecule's
                adjacency matrix

        Returns:
            ~chainer.Variable: minibatch of fingerprint
        """

        if self.G is None:
            if isinstance(adj, Variable):
                adj = adj.data
            A = (adj != 0).astype(np.int32)
            self.G = Graph(A[0])

        self.report_singular_values()

        activations = []

        if graph.dtype == self.xp.int32:
            # atom_array: (minibatch, nodes)
            h = self.embed(graph)
        else:
            h = self.compress(graph)

        if self.unchain:
            h.unchain_backward()

        activations.append(h)

        # h: (minibatch, nodes, ch)
        if isinstance(adj, Variable):
            w_adj = adj.data
        else:
            w_adj = adj
        w_adj = Variable(w_adj, requires_grad=False)

        # --- RSGCN update ---
        for i, (gconv, bnorm) in enumerate(zip(self.gconvs,
                                               self.bnorms)):
            h = gconv(h, w_adj)
            activations.append(h)

            if bnorm is not None:
                h = bnorm(h)
            if self.dropout_ratio > 0.:
                h = functions.dropout(h, ratio=self.dropout_ratio)
            if i < self.n_layers - 1:
                h = functions.relu(h)

        if self.no_readout:
            y = h
        else:
            # --- readout ---
            y = self.readout(h)

        d_perp, d_parallel = _compute_distance_to_invariant_space(
            self.G, activations)
        self.report_distance_to_invariant_space(d_perp, d_parallel)

        if return_all_activations:
            return y, activations
        else:
            return y

    def normalize(self, s0=1.):
        for gconv in self.gconvs:
            w = gconv.graph_linear.W.array
            s = _compute_singular_value(w)
            w[:] = w / s * s0

    def report_singular_values(self):
        for i, gconv in enumerate(self.gconvs):
            w = gconv.graph_linear.W.array
            s = _compute_singular_value(w)
            reporter.report({'s_value_{}'.format(i): s})

    def report_distance_to_invariant_space(self, d_perp, d_parallel):
        for i, (d1, d2) in enumerate(zip(d_perp, d_parallel)):
            reporter.report({'d_m_perp_{}'.format(i): d1})
            reporter.report({'d_m_parallel_{}'.format(i): d2})
            reporter.report({'tangent_{}'.format(i): d1 / d2})

    def get_report_targets(self):
        singular_values = ['s_value_{}'.format(i)
                           for i in range(len(self.gconvs))]
        distance_perp = ['d_m_perp_{}'.format(i)
                         for i in range(len(self.gconvs))]
        distance_parallel = ['d_m_parallel_{}'.format(i)
                             for i in range(len(self.gconvs))]
        tangent = ['tangent_{}'.format(i)
                   for i in range(len(self.gconvs))]
        return (singular_values + distance_perp
                + distance_parallel + tangent)


def _compute_singular_value(W):
    xp = chainer.backend.get_array_module(W)
    return xp.linalg.svd(W)[1][0]


class Graph(object):
    def __init__(self, A):
        self.xp = chainer.backend.get_array_module(A)
        assert A.shape == (len(A), len(A))
        self.xp.testing.assert_array_equal(A, A.T)

        self.A = A
        self.D = A.sum(axis=1)
        A_cpu = A if self.xp is np else A.get()
        G = nx.from_numpy_matrix(A_cpu)
        self.connected_components = list(nx.connected_components(G))

    def perpendicular_component(self, X):
        """Extracts the component perpendicular to the invariant space.

           The invariant space is U otimes R^C where
           U is the eigenspace associated with the smallest eigenvalue
           of the augmented normalized Laplacian.
           Specifically, U has the orthonormal basis (e_i)_{i=1, ..., A}
           defined by e_i = tilde{D}delta_i.
           Here, A is the number of connected components,
           tilde{D}:=D + I, delta_{ij} = 1 if node j belongs to
           the i-th connected component.
        """

        assert len(X) == len(self.A)
        # e is the orthogonal basis of the eigenspace.
        # Use Gram-Schmidt
        X_orig = X
        for indices in self.connected_components:
            # indices = list(indices) did not work
            # because the type of elements behaves strangely:
            # type(indices[0]) #=> <class 'int'>
            # type(indices[1]) #=> <class 'numpy.int64'>
            # type(indices[2]) #=> <class 'numpy.int64'>
            indices = [int(i) for i in list(indices)]
            e = self.xp.zeros(len(self.A), dtype=X.dtype)
            e[indices] = self.xp.sqrt(self.D[indices] + 1)
            e_norm = self.xp.linalg.norm(e)
            inner_prod = self.xp.dot(e, X)
            X = X - self.xp.outer(e, inner_prod) / e_norm ** 2
        X_perp = X
        X_parallel = X_orig - X
        return X_perp, X_parallel

    def compute_distance(self, X):
        X_perp, X_parallel = self.perpendicular_component(X)
        d_perp = self.xp.linalg.norm(X_perp)
        d_parallel = self.xp.linalg.norm(X_parallel)
        if self.xp is not np:
            d_perp = d_perp.get()
            d_parallel = d_parallel.get()
        return d_perp, d_parallel


def _compute_distance_to_invariant_space(G, ys):
    ds = [G.compute_distance(y.array[0]) for y in ys]
    d_perp = np.array([d[0] for d in ds])
    d_parallel = np.array([d[1] for d in ds])
    return d_perp, d_parallel
