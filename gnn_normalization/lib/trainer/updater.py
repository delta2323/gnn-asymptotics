# " -*- coding: utf-8 -*-"
# ------------------------------------------------------------------------------
# Name:        semisupervised_nodeclassification_updater
# Purpose:
#
#              inputs:
#
#              outputs:
#
# Author:      Katsuhiko Ishiguro <ishiguro@preferred.jp>  # NOQA
# License:     All rights reserved unless specified.
# Created:     20/03/2019 (DD/MM/YY)
# Last update: 28/03/2019 (DD/MM/YY)
# ------------------------------------------------------------------------------

import chainer


class Updater(chainer.training.StandardUpdater):
    def __init__(self, model, iterator,
                 optimizer, device, normalize):
        self.model = model
        self.iterator = iterator
        self.optimizer = optimizer
        self.normalize = normalize
        super(Updater, self).__init__(iterator=iterator,
                                      optimizer=optimizer,
                                      device=device)

    def update_core(self):
        optimizer = self.get_optimizer('main')
        dataset = self.get_iterator('main').dataset
        train_loss = self.model(dataset)

        self.model.cleargrads()
        train_loss.backward()
        optimizer.update()

        if self.normalize > 0 and hasattr(self.model, 'normalize'):
            self.model.normalize(self.normalize)

        chainer.reporter.report(
            {'train_loss': train_loss,
             'val_loss': self.model.loss['val'],
             'test_loss': self.model.loss['test'],
             'acc_train': self.model.acc['train'],
             'acc_val': self.model.acc['val'],
             'acc_test': self.model.acc['test']})
