import torch
from torch import nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn.functional as nnf

class Model(nn.Module):
    def __init__(self, **kwargs):
        super(Model, self).__init__()

    def forward(self, x, **kwargs):
        return x

    @staticmethod
    def create_inputs():
        return {
            'x': torch.FloatTensor(1, 1),
        }

if __name__ == '__main__':
    batch_size = 2
    opts = {
    }

    net = Model(**opts)

    inp = {
        'x': Variable(torch.rand(1, 1))
    }

    loss = net(**inp)
    loss.backward()
