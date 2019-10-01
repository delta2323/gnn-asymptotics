import os

from lib.model import model as model_
from lib.model import predictor as predictor_
from lib.trainer import trainer as trainer_


def run_single(dataset, config):
    predictor = predictor_.setup_predictor(
        config['model']['type'],
        config['model']['unit'],
        config['model']['conv_layers'],
        config['model']['class_num'],
        config['model']['initial_weight_scale'],
        config['model']['train_graph_conv'],
        config['model']['unchain'])
    model = model_.setup_model(predictor, config['task']['type'])
    trainer = trainer_.Trainer(model)
    trainer_config = {'gpu': config['trainer']['gpu'],
                      'iteration': config['trainer']['iteration'],
                      'frequency': -1,
                      'optimizer': config['optimizer'],
                      'normalize': config['trainer']['normalize'],
                      'out': config['trainer']['out']}
    trainer.setup(dataset, trainer_config)
    trainer.run()

    train_acc = float(trainer.model.acc['train'].array)
    val_acc = float(trainer.model.acc['val'].array)
    test_acc = float(trainer.model.acc['test'].array)
    return train_acc, val_acc, test_acc


def run(dataset, config, training_per_trial):
    N = training_per_trial
    train_acc, val_acc, test_acc = 0., 0., 0.
    out_prefix = config['trainer']['out']
    for i in range(N):
        config['trainer']['out'] = os.path.join(out_prefix, str(i))
        ret = run_single(dataset, config)
        print(ret)
        train_acc += ret[0]
        val_acc += ret[1]
        test_acc += ret[2]
    train_acc /= N
    val_acc /= N
    test_acc /= N
    return train_acc, val_acc, test_acc
