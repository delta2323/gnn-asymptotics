from chainer import iterators
from chainer import optimizers
from chainer import training
from chainer.training import extensions as E

from . import updater as updater_


class Trainer(object):

    def __init__(self, model):
        self.model = model

    def setup(self, dataset, config):
        if config['gpu'] >= 0:
            dataset.to_gpu(config['gpu'])

        train_iter = iterators.SerialIterator(dataset, 1)

        optimizer_type = config['optimizer']['type']
        learning_rate = config['optimizer']['learning_rate']
        if optimizer_type == 'SGD':
            optimizer = optimizers.SGD(learning_rate)
        elif optimizer_type == 'MomentumSGD':
            optimizer = optimizers.MomentumSGD(learning_rate)
        elif optimizer_type == 'Adam':
            optimizer = optimizers.Adam(learning_rate)
        else:
            raise ValueError('Unexpected optimizer name:{}'.format(
                    optimizer_type))
        optimizer.setup(self.model)

        updater = updater_.Updater(self.model,
                                   train_iter,
                                   optimizer,
                                   config['gpu'],
                                   config['normalize'])

        self.trainer = training.Trainer(updater,
                                        (config['iteration'], 'iteration'),
                                        out=config['out'])

        if config['frequency'] == -1:
            frequency = config['iteration']
        else:
            frequency = max(1, config['frequency'])

        print_report_targets = ['iteration', 'elapsed_time', 'train_loss',
                                'val_loss', 'test_loss', 'train_acc',
                                'val_acc', 'test_acc']
        print_report_targets += self.model.get_report_targets()
        self.trainer.extend(E.PrintReport(print_report_targets),
                            trigger=(1, 'iteration'))
        self.trainer.extend(E.LogReport(keys=print_report_targets,
                                        trigger=(1, 'iteration')))

    def run(self):
        if getattr(self, 'trainer', None) is None:
            raise Exception('Trainer is not setup. '
                            'Call Trainer.setup() before.')
        self.trainer.run()
