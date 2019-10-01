import numpy as np


def make_sample_points(L, T, N, C):
    x = [np.linspace(-L, L, T) for _ in range(N * C)]
    p = np.meshgrid(*x)
    p = [p_.ravel() for p_ in p]
    p = np.stack(p, axis=-1)
    p = p.reshape(-1, N, C)
    return p
