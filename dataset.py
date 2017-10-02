"""A Dataset that loads the dataset."""

import torch
import torch.utils.data

import common


class Dataset(torch.utils.data.Dataset):
    """Loads the data."""

    def __init__(self, dataset, part, **unused_kwargs):
        super(Dataset, self).__init__()

        self.part = part

        self.samples = common.unpickle(f'{dataset}/{part}.pkl')

    def __getitem__(self, index):
        inputs, outputs = self.samples[index]

        return {
            'inputs': inputs,
            'outputs_tgt': outputs,
        }

    def __len__(self):
        return len(self.samples)


def create(*args, **kwargs):
    """Returns a Dataset."""
    return Dataset(*args, **kwargs)


def _test_dataset():
    # pylint: disable=unused-variable
    dataset = 'data/dataset'
    part = 'test'
    debug = True

    ds_test = Dataset(**locals())
    datum = ds_test[0]

    for i in torch.randperm(len(ds_test))[:1000]:
        datum = ds_test[i]


if __name__ == '__main__':
    _test_dataset()
