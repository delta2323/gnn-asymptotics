import numpy as np


def make_dynamics(P, W, b):
    def f(x):
        y = np.tensordot(P, x, axes=(1, 1))
        y = y.transpose((1, 0, 2))
        x_ = np.matmul(y, W) + b
        x_ = np.where(x_ > 0, x_, 0)
        return x_
    return f
