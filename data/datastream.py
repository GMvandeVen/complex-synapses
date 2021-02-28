import itertools
import torch
from torch.utils.data import DataLoader



def repeater(data_loader):
    '''Function to enable looping through a data-loader indefinetely.'''
    for loader in itertools.repeat(data_loader):
        for data in loader:
            yield data


class DataStream:
    '''Iterator for setting up data-stream, with classes or tasks/domains given by `label_stream`.'''

    def __init__(self, datasets, label_stream, batch=1, per_batch=False, shuffle=True, return_task=False):
        '''Instantiate the DataStream-object.

        Args:
            datasets (list): list of Datasets, one for each label
            label_stream (LabelStream): iterator dictating from which label (task, domain or class) to sample
            batch (int, optional): # of samples per batch (default: ``1``)
            per_batch (bool, optional): if ``True``, each label from `label_stream` specifies entire batch;
                if ``False``, there is separate label for each sample in batch (default: ``False``)
            shuffle (bool, optional): whether the DataLoader should shuffle the Datasets (default: ``True``)
            return_task (bool, optional): whether identity of the task/domain should be returned (default: ``False``)
        '''
        
        self.datasets = datasets
        self.label_stream = label_stream
        self.n_labels = label_stream.n_labels
        self.batch = batch
        self.per_batch = per_batch
        self.return_task = return_task

        # To keep track of the actual label-sequence being used
        self.sequence = []

        # Create separate data-loader for each label (using 'repeater' to enable looping through them indefinitely)
        self.dataloaders = []
        for label in range(self.n_labels):
            self.dataloaders.append(repeater(
                DataLoader(datasets[label], batch_size=batch if per_batch else 1, shuffle=shuffle, drop_last=True)
            ))

    def __iter__(self):
        return self

    def __next__(self):
        '''Function to return the next batch (x,y,t).'''
        if self.per_batch or self.batch==1:
            # All samples in the batch come from same label.
            label = next(self.label_stream)
            self.sequence.append(label)
            (x, y) = next(self.dataloaders[label])
            t = label+1 if self.return_task else None
        else:
            # Multiple samples per batch come from different labels.
            x = []
            y = []
            t = [] if self.return_task else None
            for batch_id in range(self.batch):
                label = next(self.label_stream)
                self.sequence.append(label)
                (xi, yi) = next(self.dataloaders[label])
                x.append(xi)
                y.append(yi)
                if self.return_task:
                    t.append(label+1)
            x = torch.cat(x)
            y = torch.cat(y)
        return (x,y,t)
