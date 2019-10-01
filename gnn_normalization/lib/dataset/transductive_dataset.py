from chainer import cuda
from chainer.dataset import dataset_mixin


class TransductiveDataset(dataset_mixin.DatasetMixin):
    def __init__(self, X, Adense, labels, indices):
        self.X = X
        self.A = Adense
        self.labels = labels
        self.train_idx, self.val_idx, self.test_idx = indices

    def to_gpu(self, device=None):
        self.X = cuda.to_gpu(self.X, device=device)
        self.A = cuda.to_gpu(self.A, device=device)
        self.labels = cuda.to_gpu(self.labels, device=device)
        self.train_idx = cuda.to_gpu(self.train_idx, device=device)
        self.val_idx = cuda.to_gpu(self.val_idx, device=device)
        self.test_idx = cuda.to_gpu(self.test_idx, device=device)

    def __len__(self):
        return len(self.X)

    def get_example(self, i):
        return None
