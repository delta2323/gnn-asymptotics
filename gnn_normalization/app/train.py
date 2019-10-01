#!/usr/bin/env python
import argparse
import json
import os

import numpy as np
import optuna

from lib.dataset import dataset as dataset_
from lib.run import run as run_


def _objective(trial, dataset, config,
               training_per_trial):
    unit = int(trial.suggest_discrete_uniform(
            'unit', 10, config['iteration'], 10))
    conv_layers = trial.suggest_int(
        'conv_layers', config['conv_layers'], config['conv_layers'])
    iteration = int(trial.suggest_discrete_uniform(
            'iteration', 10, config['iteration'], 10))
    optimizer = trial.suggest_categorical(
        'optimizer', ['SGD', 'MomentumSGD', 'Adam'])
    learning_rate = trial.suggest_loguniform(
        'learning_rate', 1e-5, config['learning_rate'])

    out = os.path.join(config['out'], str(trial.number))
    trial_config = {'task': {'type': config['task_type']},
                    'model': {'type': 'rsgcn',
                              'unit': unit,
                              'conv_layers': conv_layers,
                              'class_num': config['class_num'],
                              'initial_weight_scale':
                              config['initial_weight_scale'],
                              'train_graph_conv':
                              config['train_graph_conv'],
                              'unchain': config['unchain']},
                    'trainer': {'out': out,
                                'iteration': iteration,
                                'normalize': config['normalize'],
                                'gpu': config['gpu']},
                    'optimizer': {'type': optimizer,
                                  'learning_rate': learning_rate}}

    train_acc, val_acc, test_acc = run_.run(dataset, trial_config,
                                            training_per_trial)

    trial.set_user_attr('train_acc', train_acc)
    trial.set_user_attr('val_acc', val_acc)
    trial.set_user_attr('test_acc', test_acc)
    print(trial.params)
    print('val_acc:{}, test_acc:{}'.format(val_acc, test_acc))
    return -val_acc


def parse_arguments():
    parser = argparse.ArgumentParser(description='molnet example')
    # Weight
    parser.add_argument('--normalize', '-n', type=float, default=-1,
                        help='If true, we normalize weight '
                        'at every iteration.')
    parser.add_argument('--initial-weight-scale', '-I', type=float,
                        default=None,
                        help='Maximum singular values of '
                        'initial weights of graph convolution. '
                        'If None, we do not normalize weights.')
    parser.add_argument('--untrain-graph-conv', '-U', action='store_true',
                        help='If True, we do not train graph '
                        'convolution layer of RSGCN and leave '
                        'it randomly initialized values.')
    parser.add_argument('--unchain', '-C', action='store_true',
                        help='If True, we do not train the first compress '
                        'layer of RSGCN and leave it randomly '
                        'initialized values.')
    # Dataset
    parser.add_argument('--dataset', '-d', type=str,
                        choices=['citeseer', 'noisy-citeseer',
                                 'cora', 'noisy-cora',
                                 'pubmed', 'noisy-pubmed'],
                        default='citeseer',
                        help='name of the dataset that training is run on')
    # Optimization parameters
    parser.add_argument('--trial', '-t', type=int, default=100,
                        help='trial')
    # Hyperparameters
    parser.add_argument('--conv-layers', '-c', type=int, default=5,
                        help='maximum conv layers')
    parser.add_argument('--unit', '-u', type=int, default=300,
                        help='maximum unit size')
    parser.add_argument('--iteration', '-i', type=int, default=200,
                        help='maximum iteration')
    parser.add_argument('--learning-rate', '-l', type=float, default=1e-2,
                        help='maximum learning rate')
    # Other parameters
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='id of gpu to use; negative value means running'
                        'the code on cpu')
    parser.add_argument('--training-per-trial', '-p', type=int, default=3,
                        help='# of trainings per trial.')
    parser.add_argument('--seed', '-s', type=int, default=0,
                        help='Random seed')
    parser.add_argument('--out', '-o', type=str, default='results',
                        help='Path to result directory')
    return parser.parse_args()


def main():
    args = parse_arguments()
    np.random.seed(args.seed)

    dataset, task_type, class_num = dataset_.get_dataset(
        args.dataset)

    if args.gpu >= 0:
        import cupy as cp
        cp.cuda.Device(args.gpu).use()
        cp.random.seed(args.seed)

    study = optuna.create_study(study_name='test')
    config = {'conv_layers': args.conv_layers,
              'unit': args.unit,
              'iteration': args.iteration,
              'class_num': class_num,
              'task_type': task_type,
              'out': args.out,
              'learning_rate': args.learning_rate,
              'normalize': args.normalize,
              'gpu': args.gpu,
              'initial_weight_scale': args.initial_weight_scale,
              'train_graph_conv': not args.untrain_graph_conv,
              'unchain': args.unchain}

    def objective(trial):
        return _objective(trial, dataset, config,
                          args.training_per_trial)

    study.optimize(objective, n_trials=args.trial)
    print('Best Parameter', study.best_trial.params)
    best_trial_info = {
        'params': study.best_trial.params,
        'number': study.best_trial.number
    }
    with open(os.path.join(args.out, 'best_trial.json'), 'w') as f:
        json.dump(best_trial_info, f)

    acc = {
        'train': study.best_trial.user_attrs['train_acc'],
        'val': study.best_trial.user_attrs['val_acc'],
        'test': study.best_trial.user_attrs['test_acc']
    }
    with open(os.path.join(args.out, 'accuracies.json'), 'w') as f:
        json.dump(acc, f)
    print('Train Accuracy: {}'.format(acc['train']))
    print('Val Accuracy: {}'.format(acc['val']))
    print('Test Accuracy: {}'.format(acc['test']))


if __name__ == '__main__':
    main()
