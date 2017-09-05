"""The model."""

import torch
from torch import nn
from torch.autograd import Variable


class Model(nn.Module):
    """An `nn.Module` representing the model."""

    def __init__(self, **unused_kwargs):
        super(Model, self).__init__()

    def forward(self, inputs, **unused_kwargs):
        return inputs.sum()

    @staticmethod
    def create_inputs():
        """Returns a dict of tensors that this model may use as input."""
        return {
            'inputs': torch.FloatTensor(),
        }


if __name__ == '__main__':
    batch_size = 2
    opts = {
        'debug': True,
    }

    model = Model(**opts)

    inp = {
        'x': torch.rand(batch_size, 3, 224, 224),
    }
    for name, tensor in inp.items():
        inp[name] = Variable(tensor)

    loss = model(**inp)
    loss.backward()
