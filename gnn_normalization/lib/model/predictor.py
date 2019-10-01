from chainer_chemistry import links

from . import graph_conv_predictor as g_predictor
from . import rsgcn


def setup_predictor(method, n_unit, conv_layers, class_num,
                    initial_weight_scale, train_graph_conv,
                    unchain):
    """Sets up the predictor"""

    if method == 'rsgcn':
        model = rsgcn.RSGCN(out_dim=n_unit,
                            hidden_dim=n_unit,
                            n_layers=conv_layers,
                            no_readout=True,
                            dropout_ratio=0.,
                            initial_weight_scale=initial_weight_scale,
                            unchain=unchain)
    else:
        raise ValueError('[ERROR] Invalid method: {}'.format(method))

    graphout = links.GraphLinear(out_size=class_num, in_size=n_unit)

    return g_predictor.GraphConvPredictor(
        model, graphout,
        train_graph_conv=train_graph_conv)
