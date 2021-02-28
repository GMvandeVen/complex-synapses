import numpy as np
import torch
import utils


####--------------------------------------------------------------------------------------------------------------####

####-----------------------------####
####----CLASSIFIER EVALUATION----####
####-----------------------------####


def validate(model, dataset, task=None, batch_size=128, test_size=1024, shuffle=True, verbose=True):
    '''Evaluate precision (= accuracy or proportion correct) of a classifier-model on given dataset.

    Args:
        model (Classifier): model to be evaluated
        dataset (Dataset): from which to draw samples to evaluate accuracy of `model`
        task (int or None, optional): if provided, evaluation is according to Task-IL scenario (default: None)
        batch_size (int, optional): number of samples to evaluate per forward pass (default: ``128``)
        test_size (int or None, optional): maximum total number of samples to evaluate (default: ``1024``)
        shuffle (bool, optional): whether data should be shuffled in the DataLoader (default: ``True``)
        verbose (bool, optional): whether output should be printed to screen (default: ``True``)
    '''

    # Set model to eval()-mode.
    model.eval()

    # Loop over batches in [dataset].
    data_loader = utils.get_data_loader(dataset, batch_size, cuda=model._is_on_cuda(), shuffle=shuffle)
    total_tested = total_correct = 0
    for data, labels in data_loader:
        # -break on [test_size] (if "None", full dataset is used)
        if test_size:
            if total_tested >= test_size:
                break
        # -evaluate model (if requested, only on [allowed_classes])
        data, labels = data.to(model._device()), labels.to(model._device())
        with torch.no_grad():
            scores = model(data, task=task)
            _, predicted = torch.max(scores, 1)
        # -update statistics
        total_correct += (predicted == labels).sum().item()
        total_tested += len(data)
    precision = total_correct / total_tested

    # Print result on screen (if requested) and return it.
    if verbose:
        print('=> precision: {:.3f}'.format(precision))
    return precision


def initiate_metrics_dict(n_labels, classes=False):
    '''Initiate <dict> with all measures to keep track of.'''
    key = "class" if classes else "task"
    metrics_dict = {}
    metrics_dict["acc_per_{}".format(key)] = {}
    for i in range(n_labels):
        metrics_dict["acc_per_{}".format(key)]["{}_{}".format(key, i+1)] = []   # accuracy per task / class
    metrics_dict["ave_acc"] = []                                   # average accuracy (over all tasks / classes)
    metrics_dict["ave_acc_so_far"] = []                            # average acuracy (over all tasks / classes so far)
    metrics_dict["iters"] = []                                     # total number of iterations so far
    return metrics_dict


def precision(model, datasets, up_to=None, provide_task_info=False, test_size=None, shuffle=True, verbose=False):
    '''Evaluate precision of classifier on all entries of `datasets` (= list of Datasets) up to `up_to` (= int).'''

    # Evaluate accuracy of model predictions for all tasks/domains/classes so far (reporting "0" for future ones).
    precs = []
    for i in range(len(datasets)):
        if (up_to is None) or (i+1 <= up_to):
            precs.append(validate(model, datasets[i], test_size=test_size, shuffle=shuffle, verbose=verbose,
                                  task=i+1 if provide_task_info else None))
        else:
            precs.append(0)
    average_precs = np.mean(precs) if up_to is None else sum([precs[task_id] for task_id in range(up_to)]) / up_to
    # Print results on screen.
    if verbose:
        print(' => ave precision: {:.3f}'.format(average_precs))

    return (precs, average_precs)
