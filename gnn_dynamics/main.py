import argparse

import numpy as np

import dynamics
import plot
import random_matrix
import sample_point


parser = argparse.ArgumentParser(description='molnet example')
parser.add_argument('--seed', '-s', type=int,
                    help='random seed', default=0)
parser.add_argument('--node-size', '-N', type=int,
                    help='Node size', default=2)
parser.add_argument('--channel-size', '-C', type=int,
                    help='Node size', default=1)
parser.add_argument('--largest-singular-value', '-S', type=float,
                    help='The largest singular value', default=1.)
parser.add_argument('--sample-point-size', '-T', type=int,
                    help='# of Sample points per an interval', default=20)
parser.add_argument('--length', '-L', type=float,
                    help='Length of each side of domain', default=1.)
parser.add_argument('--out', '-O', type=str,
                    help='Output directory', default='result')
args = parser.parse_args()

np.random.seed(args.seed)


# Generate random matrices
N = args.node_size
lambda_ = 0.5 * np.ones(N)
lambda_[0] = 1.0
P = random_matrix.make_p(lambda_)

C = args.channel_size
s = np.ones(C) * -0.5
s[0] = args.largest_singular_value
W = random_matrix.make_w(s)
b = np.zeros(C)

# Make dynamics
f = dynamics.make_dynamics(P, W, b)

# Make sample points and forward one time step
T = args.sample_point_size
L = args.length
p = sample_point.make_sample_points(L, T, N, C)
p_next = f(p)


# Debug print
lambda_, e = np.linalg.eig(P)
e = e[:, np.argsort(lambda_)]
e1 = e[:, -1]  # eigen vector for largest eigen vector
e2 = e[:, -2]  # eigen vector for second largest eigen vector
_, s, _ = np.linalg.svd(W)
print('P: ', P)
print('W:', W)
print('Singular values: ', s)
print('Largest eigen vector 1: ', e1)
print('Second Largest eigen vector: ', e2)


plot.streamplot(p, p_next, L, e1[1] / e1[0],
                'W={}'.format(float(W)),
                args.out)
