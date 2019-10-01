from . import classifier


def setup_model(predictor, task_type):
    if task_type == 'classification':
        return classifier.Classifier(predictor)
    elif task_type == 'regression':
        return NotImplementedError
    else:
        raise ValueError('No such task_type: {}'.format(task_type))
