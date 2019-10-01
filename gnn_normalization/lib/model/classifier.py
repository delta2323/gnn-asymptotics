import chainer
import chainer.functions as F
from chainer import reporter


def compute_loss(logit, labels):
    return F.softmax_cross_entropy(logit, labels)


def compute_acc(logit, labels):
    return F.accuracy(logit, labels)


class Classifier(chainer.Chain):

    def __init__(self, predictor):
        super(Classifier, self).__init__()
        with self.init_scope():
            self.predictor = predictor

    def compute_loss(self, logit, dataset):
        labels = F.argmax(dataset.labels, axis=1)
        train_loss = compute_loss(logit[dataset.train_idx],
                                  labels[dataset.train_idx])
        val_loss = compute_loss(logit[dataset.val_idx],
                                labels[dataset.val_idx])
        test_loss = compute_loss(logit[dataset.test_idx],
                                 labels[dataset.test_idx])
        return {'train': train_loss,
                'val': val_loss,
                'test': test_loss}

    def compute_acc(self, logit, dataset):
        labels = F.argmax(dataset.labels, axis=1)
        train_acc = compute_acc(logit[dataset.train_idx],
                                labels[dataset.train_idx])
        val_acc = compute_acc(logit[dataset.val_idx],
                              labels[dataset.val_idx])
        test_acc = compute_acc(logit[dataset.test_idx],
                               labels[dataset.test_idx])
        return {'train': train_acc,
                'val': val_acc,
                'test': test_acc}

    def _forward(self, dataset):
        logit = self.predictor(dataset.X, dataset.A)
        return logit[0]

    def forward(self, dataset):
        logit = self._forward(dataset)
        train_mode_loss = self.compute_loss(logit, dataset)
        train_mode_acc = self.compute_acc(logit, dataset)

        with chainer.using_config('train', False):
            with chainer.using_config('enable_backprop', False):
                logit = self._forward(dataset)
        test_mode_loss = self.compute_loss(logit, dataset)
        test_mode_acc = self.compute_acc(logit, dataset)

        self.loss = {'train': train_mode_loss['train'],
                     'val': test_mode_loss['val'],
                     'test': test_mode_loss['test']}
        self.acc = {'train': train_mode_acc['train'],
                    'val': test_mode_acc['val'],
                    'test': test_mode_acc['test']}

        reporter.report({'train_loss': self.loss['train'],
                         'val_loss': self.loss['val'],
                         'test_loss': self.loss['test'],
                         'train_acc': self.acc['train'],
                         'val_acc': self.acc['val'],
                         'test_acc': self.acc['test']})

        return self.loss['train']

    def normalize(self, s0=1.):
        self.predictor.normalize(s0)

    def get_report_targets(self):
        return self.predictor.get_report_targets()
