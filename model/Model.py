"""The model."""

import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as nnf


class Model(nn.Module):
    """An `nn.Module` representing the model."""

    def __init__(self, **unused_kwargs):
        super(Model, self).__init__()

        self.emb = nn.Sequential(
            nn.Linear(4, 2),
            nn.LogSoftmax(),
        )

    def forward(self, inputs, outputs_tgt=None, **unused_kwargs):
        outputs_preds = self.emb(inputs)

        has_target = False
        for var_name, var_val in locals().items():
            has_target = (has_target or
                          (var_name.endswith('_tgt') and var_val is not None))
        if has_target:
            outputs_loss = nnf.nll_loss(outputs_preds, outputs_tgt)

        preds = {}
        losses = {}
        for var_name, var in locals().items():
            if var_name.endswith('_loss'):
                losses[var_name.replace('_loss', '')] = var
            elif var_name.endswith('_preds'):
                preds[var_name.replace('_preds', '')] = var

        return {
            'losses': losses,
            'preds': preds,
        }

    def create_inputs(self):
        """Returns a dict of tensors that this model may use as input."""
        return {
            'inputs': torch.FloatTensor(),
            'outputs_tgt': torch.LongTensor(),
        }


def _test_model():
    batch_size = 2
    debug = True

    model = Model(**locals())

    inp = {
        'inputs': torch.rand(batch_size, 4).log(),
        'outputs_tgt': torch.LongTensor(batch_size).random_(2),
    }
    for name, tensor in inp.items():
        inp[name] = Variable(tensor)

    outputs = model(**inp)
    if outputs['losses']:
        sum(outputs['losses'].values()).backward()


if __name__ == '__main__':
    _test_model()
