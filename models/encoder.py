import torch
from torch import nn
from torch.nn import functional as F
from models.layers import fc_layer
from models.nets import MLP
import models.modules as modules


class Classifier(nn.Module):
    '''Model for classifying images.'''

    def __init__(self, image_size, image_channels, classes, fc_layers=3, fc_units=1000):

        # Configurations.
        super().__init__()
        self.classes = classes
        self.label = "Classifier"
        self.fc_layers = fc_layers

        # Attributes that need to be set before training.
        self.optimizer = None
        self.multi_head = None   #-->  <list> with for each task its class-IDs

        # Check whether there is at least 1 fc-layer.
        if fc_layers<1:
            raise ValueError("The classifier needs to have at least 1 fully-connected layer.")

        ######------SPECIFY MODEL------######

        # Flatten image to 2D-tensor.
        self.flatten = modules.Flatten()

        # Fully connected hidden layers.
        self.fcE = MLP(input_size=image_channels*image_size**2, output_size=fc_units, layers=fc_layers-1,
                       hid_size=fc_units)
        self.mlp_output_size = fc_units if fc_layers>1 else image_channels*image_size**2

        # Classifier.
        self.classifier = fc_layer(self.mlp_output_size, classes, nl='none', bias=True)


    def _device(self):
        return next(self.parameters()).device

    def _is_on_cuda(self):
        return next(self.parameters()).is_cuda


    def list_init_layers(self):
        '''Return list of modules whose parameters could be initialized differently (i.e., conv- or fc-layers).'''
        list = []
        list += self.fcE.list_init_layers()
        list += self.classifier.list_init_layers()
        return list

    @property
    def name(self):
        if self.fc_layers>1:
            return "{}_c{}".format(self.fcE.name, self.classes)
        else:
            return "i{}_c{}".format(self.mlp_output_size, self.classes)

    def forward(self, x, task=None):
        # Run forward pass of model.
        final_features = self.fcE(self.flatten(x))
        logits = self.classifier(final_features)
        # If using multi-headed output layer, select the correct "output-head" depending on provided task.
        # --> [task]:  <int> task-ID (if all samples in [x] from same task)  -OR-
        #              <list>/<tensor> with task-IDs of all samples in [x]
        if task is None or self.multi_head is None:
            return logits
        elif type(task)==int:
            return logits[:, self.multi_head[task-1]]
        else:
            task_indeces = []
            for i in task:
                task_indeces.append(self.multi_head[i-1])
            return logits.gather(dim=1, index=torch.LongTensor(task_indeces).to(self._device()))


    def feature_extractor(self, images):
        return self.fcE(self.flatten(images))


    def train_a_batch(self, x, y, task=None):
        '''Train model for one batch (`x`, `y`, `task`).

        Args:
            x (tensor): batch of inputs
            y (tensor): batch of corresponding labels
            task (int, list/tensor or None): taskID of entire batch (if int) or of each sample in batch (if list/tensor)
        '''

        # Set model to training-mode
        self.train()

        # Reset optimizer
        self.optimizer.zero_grad()

        # Run model
        y_hat = self(x, task=task)

        # Calculate training-precision
        _, predicted = torch.max(y_hat, 1)
        correct = (y == predicted)
        precision = None if y is None else correct.sum().item() / x.size(0)

        # Get "gradients" required for updates (i.e. the "delta-Ws" or "desired updates")
        # -calculate prediction loss
        loss = F.cross_entropy(input=y_hat, target=y, reduction='mean')
        # -backpropagate errors
        loss.backward()

        # Take optimization-step
        self.optimizer.step()

        # Return the dictionary with training-loss and training-precision
        return {
            'loss': loss.item(),
            'precision': precision if precision is not None else 0.,
        }

