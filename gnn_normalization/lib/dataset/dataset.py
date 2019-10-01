from lib.dataset import citation
from lib.dataset import reddit


def get_dataset(name, normalize=True):
    if name == 'citeseer' or name == 'noisy-citeseer':
        noisy = name == 'noisy-citeseer'
        all_data = citation.load(
            'citeseer', normalize, noisy)
        task_type = 'classification'
    elif name == 'cora' or name == 'noisy-cora':
        noisy = name == 'noisy-cora'
        all_data = citation.load(
            'cora', normalize, noisy)
        task_type = 'classification'
    elif name == "pubmed" or name == 'noisy-pubmed':
        noisy = name == 'noisy-pubmed'
        all_data = citation.load(
            'pubmed', normalize, noisy)
        task_type = 'classification'
    elif name == 'reddit':
        raise NotImplementedError
        all_data = reddit.load_reddit_data(
            name, normalize)
        task_type = 'regression'
    else:
        raise ValueError("no such dataset is defined: " + name)

    if name == 'cora' or name == 'noisy-cora':
        class_num = 7
    elif name == 'citeseer' or name == 'noisy-citeseer':
        class_num = 6
    elif name == 'pubmed' or name == 'noisy-pubmed':
        class_num = 3
    elif name == 'reddit':
        class_num = 1

    return all_data, task_type, class_num
