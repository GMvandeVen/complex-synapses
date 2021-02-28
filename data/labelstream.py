import numpy as np
import random



class LabelStream:
    '''Base class for iterators that determine from which label (either task, domain or class) should be sampled.'''

    def __init__(self):
        self.n_labels = None
        self.n_tasks = None

    def __iter__(self):
        return self

    def __next__(self):
        raise NotImplementedError


class TaskBasedStream(LabelStream):
    '''Set up a label-stream with strictly separated tasks / episodes.'''

    def __init__(self, n_tasks, iters_per_task, labels_per_task=1):
        '''Instantiate the dissociated-stream object by defining its parameters.

        Args:
            n_tasks (int): number of tasks
            iters_per_task (int): number of iterations to generate per task (i.e., number of labels to pick)
            labels_per_task (int, optional): number of different labels to choose from per task (default: ``1``)
        '''

        super().__init__()
        self.n_tasks = n_tasks
        self.iters_per_task = iters_per_task
        self.labels_per_task = labels_per_task
        self.n_labels = n_tasks*labels_per_task

        # For keeping track of task
        self.count = 0
        self.task = 0
        # -if there is more than one label per task, list the labels in first task
        if labels_per_task>1:
            self.labels_in_task = list(range(0, self.labels_per_task))

    def __next__(self):
        # Update count and move to next task when all iterations of current task are done
        self.count += 1
        if self.count > self.iters_per_task:
            self.count = 0
            self.task += 1
            if self.task >= self.n_tasks:
                raise StopIteration
            # -if there is more than one label per task, list the labels in current task
            if self.labels_per_task > 1:
                self.labels_in_task = list(range(self.task*self.labels_per_task, (self.task+1)*self.labels_per_task))
        # Sample label from all labels in current task (if required) and return
        next_label = self.task if self.labels_per_task==1 else random.sample(self.labels_in_task, k=1)[0]
        return next_label


class RandomStream(LabelStream):
    '''Set up a completely random label-stream.'''

    def __init__(self, labels, n_times=1):
        super().__init__()
        self.n_labels = labels
        self.n_tasks = None
        self.n_times = n_times

    def __next__(self):
        if self.n_times>1:
            return np.random.randint(0, self.n_labels, self.n_times)
        else:
            return random.randint(0, self.n_labels-1)
