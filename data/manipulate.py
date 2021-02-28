import numpy as np
import torch
from torch.utils.data import Dataset


class ReducedDataset(Dataset):
    '''To reduce a dataset, taking only samples corresponding to provided indeces.

    This is useful for splitting a dataset into a training and validation set.
    '''

    def __init__(self, original_dataset, indeces):
        super().__init__()
        self.dataset = original_dataset
        self.indeces = indeces

    def __len__(self):
        return len(self.indeces)

    def __getitem__(self, index):
        return self.dataset[self.indeces[index]]


class TransformedDataset(Dataset):
    '''Modify existing dataset with transform; for creating multiple MNIST-permutations w/o loading data every time.'''

    def __init__(self, original_dataset, transform=None, target_transform=None):
        super().__init__()
        self.dataset = original_dataset
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        (input, target) = self.dataset[index]
        if self.transform:
            input = self.transform(input)
        if self.target_transform:
            target = self.target_transform(target)
        return (input, target)


class DatasetFromList(Dataset):
    '''Take a <list> with elements (data, labels) as input, and turn into a Dataset-object.'''

    def __init__(self, dataset_list):
        super().__init__()
        self.dataset = dataset_list

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]



class SubDataset(Dataset):
    '''To sub-sample a dataset, taking only those samples with label in `sub_labels`.

    After the selection of samples has been made, it is possible to transform the target-labels,
    which can be useful when doing continual learning with a fixed number of output units.
    '''

    def __init__(self, original_dataset, sub_labels, target_transform=None):
        super().__init__()
        self.dataset = original_dataset
        self.sub_indeces = []
        if type(sub_labels)==int:
            sub_labels = [sub_labels]
        for index in range(len(self.dataset)):
            if hasattr(original_dataset, "targets"):
                if self.dataset.target_transform is None:
                    label = self.dataset.targets[index]
                else:
                    label = self.dataset.target_transform(self.dataset.targets[index])
            else:
                label = self.dataset[index][1]
            if label in sub_labels:
                self.sub_indeces.append(index)
        self.target_transform = target_transform
        #self.targets = self.target_transform(self.dataset.targets[self.sub_indeces])
        # --> STILL TO GET TO WORK!! This would make creating subdatasets of sub-datasets substantially faster.

    def __len__(self):
        return len(self.sub_indeces)

    def __getitem__(self, index):
        sample = self.dataset[self.sub_indeces[index]]
        if self.target_transform:
            target = self.target_transform(sample[1])
            sample = (sample[0], target)
        return sample



class TemplateDataset(Dataset):
    '''Create Dataset object for a given set of ``template_patterns``, ``labels`` and ``noise_prob``.'''

    def __init__(self, template_patterns, labels, noise_prob=0.):
        '''Instantiate the TemplateDataset-object.

        Args:
            template_patterns (list): list with lists containing the "template" patterns for each class
            labels (list): list with corresponding class-labels
            noise_prob (float): probability that any given entry of the template-pattern flips when sampled
        '''
        super().__init__()
        self.template_patterns = template_patterns
        self.labels = labels
        self.length = len(template_patterns[0])
        self.noise_prob = noise_prob

    def __len__(self):
        return 1000000000000000  #-> a trick (note: creating a DataLoader with shuffling still seems to be problematic)

    def __getitem__(self, index):
        '''Samples are generated "on-the-fly" (i.e., noise is added to template-patterns at time they are accessed).'''
        label_index = np.random.choice(len(self.labels))
        label = self.labels[label_index]
        if self.noise_prob>0.:
            flip_tensor = torch.tensor(np.random.choice([-1, 1], size=self.length,
                                                        p=[self.noise_prob, 1-self.noise_prob]), dtype=torch.float)
            pattern = self.template_patterns[label_index] * flip_tensor
        else:
            pattern = self.template_patterns[label_index]
        return pattern, label


#----------------------------------------------------------------------------------------------------------#


def permutate_image_pixels(image, permutation):
    '''Permutate the pixels of an image.

    Args:
        image (3D-tensor): Original image to be permuted.
        permutation (ndarray): The new order of pixel-indeces.

    Returns:
        3D-tensor: Permuted image.
    '''

    if permutation is None:
        return image
    else:
        c, h, w = image.size()
        image = image.view(c, -1)
        image = image[:, permutation]  #--> same permutation for each channel
        image = image.view(c, h, w)
        return image
