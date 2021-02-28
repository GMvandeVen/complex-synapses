import copy
import numpy as np
from torchvision import transforms
from torch.utils.data import ConcatDataset

from data.available import AVAILABLE_TRANSFORMS, AVAILABLE_DATASETS, DATASET_CONFIGS
from data.manipulate import SubDataset, TransformedDataset, permutate_image_pixels




def get_dataset(name, type='train', download=True, capacity=None, permutation=None, dir='./datasets',
                verbose=False, target_transform=None):
    '''Load individual dataset.'''

    data_name = 'mnist' if name=='mnist28' else name
    dataset_class = AVAILABLE_DATASETS[data_name]

    # Specify image-transformations to be applied.
    dataset_transform = transforms.Compose([
        *AVAILABLE_TRANSFORMS[name], transforms.Lambda(lambda x: permutate_image_pixels(x, permutation)),
    ])

    # Load data-set.
    dataset = dataset_class('{dir}/{name}'.format(dir=dir, name=data_name), train=False if type=='test' else True,
                            download=download, transform=dataset_transform, target_transform=target_transform)

    # Print information about dataset on the screen.
    if verbose:
        print("  --> {}: '{}'-dataset consisting of {} samples".format(name, type, len(dataset)))

    # If dataset is not large enough, create copies until it is.
    if capacity is not None and len(dataset) < capacity:
        dataset_copy = copy.deepcopy(dataset)
        dataset = ConcatDataset([dataset_copy for _ in range(int(np.ceil(capacity / len(dataset))))])

    return dataset


def prepare_datasets(name, n_labels, classes=True, classes_per_task=None,
                     dir="./store/datasets", verbose=False, only_config=False, exception=False, only_test=False):
    '''Prepare training- and test-datasets for continual learning experiment.

    Args:
        name (str; `splitMNIST`|`splitCIFAR`):
        n_labels (int): number of classes or number of tasks/domains
        template_patterns (list, optional): required if ``name``=="artificial"
        noise_prob (float, optional): relevant if ``name``=="artificial",
                                        probability that entry of template-pattern is flipped (default: ``0.1``)
        classes (bool, optional): if ``True``, labels indicate classes, otherwise tasks/domains (default: ``True``)
        classes_per_task (int, optional): required if ``classes`` is ``False``
        dir (path, optional): where data is stored / should be downloaded
        verbose (bool, optional): if ``True``, print (additional) information to screen
        only_config (bool, optional): if ``True``, only return config-information (faster; data is not actually loaded)
        exception (bool, optional): if ``True``, do not shuffle labels
        only_test (bool, optional): if ``True``, only load and return test-set(s)

    Returns:
        tuple
    '''

    if name == 'splitMNIST':
        # Configurations.
        config = DATASET_CONFIGS['mnist28']
        if not only_config:
            # Prepare permutation to shuffle label-ids (to create different class batches for each random seed).
            n_class = config['classes']
            permutation = np.array(list(range(n_class))) if exception else np.random.permutation(list(range(n_class)))
            target_transform = transforms.Lambda(lambda y, x=permutation: int(permutation[y]))
            # Load train and test datasets with all classes.
            if not only_test:
                mnist_train = get_dataset('mnist28', type="train", dir=dir, target_transform=target_transform,
                                          verbose=verbose)
            mnist_test = get_dataset('mnist28', type="test", dir=dir, target_transform=target_transform,
                                     verbose=verbose)
            # Generate labels-per-task.
            labels_per_task = range(n_labels) if classes else [
                list(np.array(range(classes_per_task)) + classes_per_task*task_id) for task_id in range(n_labels)
            ]
            # Split them up into separate datasets for each task / class.
            train_datasets = []
            test_datasets = []
            for labels in labels_per_task:
                target_transform = None if classes else transforms.Lambda(lambda y, x=labels[0]: y - x)
                if (not only_test):
                    train_datasets.append(SubDataset(mnist_train, labels, target_transform=target_transform))
                test_datasets.append(SubDataset(mnist_test, labels, target_transform=target_transform))
    elif name == 'permMNIST':
        if classes:
            raise NotImplementedError("Permuted MNIST with Class-IL is not (yet) implemented.")
        # Configurations
        config = DATASET_CONFIGS['mnist']
        if not only_config:
            # Generate labels-per-task
            labels_per_task = [list(np.array(range(classes_per_task)) + classes_per_task*task_id) for task_id in range(n_labels)]
            # Prepare datasets
            train_dataset = get_dataset('mnist', type="train", dir=dir, verbose=verbose)
            test_dataset = get_dataset('mnist', type="test", dir=dir, verbose=verbose)
            # Generate permutations
            if exception:
                permutations = [None] + [np.random.permutation(config['size']**2) for _ in range(n_labels-1)]
            else:
                permutations = [np.random.permutation(config['size']**2) for _ in range(n_labels)]
            # Prepare datasets per task
            train_datasets = []
            test_datasets = []
            for task_id, perm in enumerate(permutations):
                train_datasets.append(TransformedDataset(
                    train_dataset, transform=transforms.Lambda(lambda x, p=perm: permutate_image_pixels(x, p)),
                ))
                test_datasets.append(TransformedDataset(
                    test_dataset, transform=transforms.Lambda(lambda x, p=perm: permutate_image_pixels(x, p)),
                ))


    return config if only_config else ((train_datasets, test_datasets), config, labels_per_task)