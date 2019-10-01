from lib.run import run as R
from lib.dataset import dataset as D


dataset, task_type, class_num = D.get_dataset('noisy-pubmed')
trial_config = {
    'task': {'type': task_type},
    'model': {'type': 'rsgcn',
              'unit': 70,
              'conv_layers': 9,
              'class_num': class_num,
              'initial_weight_scale': 3.0,  # to disable, set None
              'train_graph_conv': True,
              'unchain': False},
    'trainer': {'out': 'out',
                'iteration': 90,
                'normalize': 3.0,  # to disable, set -1
                'gpu': 0},
    'optimizer': {'type': 'Adam',
                  'learning_rate': 0.001578273362066244}
}

train_acc, val_acc, test_acc = R.run_single(dataset, trial_config)
print('Train Acc: {}'.format(train_acc))
print('Val Acc: {}'.format(val_acc))
print('Test Acc: {}'.format(test_acc))
