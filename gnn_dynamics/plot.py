import os

from matplotlib import pyplot as plt
import numpy as np


def _line(L, S, a):
    xs = np.linspace(-L, L, S)
    ys = xs * a
    L_eps = L - 1e-3
    idx = (xs < L_eps) & (-L_eps < xs) & (-L_eps < ys) & (ys < L_eps)
    return xs[idx], ys[idx]


def streamplot(p, p_next, L, a, title, out):
    assert p.shape[-1] == 1
    assert p.shape[-2] == 2
    p = p[..., 0]
    p_next = p_next[..., 0]
    
    delta = p_next - p
    
    n_points = len(delta)
    S = int(np.sqrt(n_points))
    assert n_points == S * S
    
    p = p.reshape(S, S, 2)
    delta = delta.reshape(S, S, 2)

    x_delta = delta[..., 0]
    y_delta = delta[..., 1]
    v = np.sqrt(x_delta ** 2 + y_delta ** 2)

    x = np.unique(p[..., 0])
    y = np.unique(p[..., 1])

    line_x, line_y = _line(5, S, a)
    plt.figure(figsize=(6, 5))
    plt.plot(line_x, line_y, c='r', linestyle=':')
    plt.xlim(x.min(), x.max())
    plt.ylim(y.min(), y.max())
    plt.title(title)
    strm = plt.streamplot(x, y, x_delta, y_delta, color=v, linewidth=1)
    plt.colorbar(strm.lines)
    plt.savefig(os.path.join(out, 'streamplot.pdf'))
