This is the code for experiments in Section 6.2, 6.3, and 6.4.

# Dependency

- chainer==6.3.0
- cuda==9.0
- cudnn==7.1
- numpy==1.17
- scipy==1.3
- optuna==0.15
- cupy-cuda90==6.3.0
- networkx==2.3
- chainer-chemistry==0.6

It is possible that we can run the code in using packages with different versions. But we do not guarantee it.

# Preparation

Place `https://github.com/tkipf/gcn/tree/master/gcn/data` as `gnn_normalization/lib/dataset/data/kipf` (e.g., `gcn/data/ind.citeseer.allx` should be copied to `gnn_normalization/lib/dataset/data/kipf/ind.citeseer.allx`.)

# Usage

Section 6.2

The result of the experiment is available in `sec_6_2.ipynb`.


Section 6.3
```
bash -x run.sh
```

Section 6.4

Add `-C` option as the argument of `app/train.py` in `run.sh`.
