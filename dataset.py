from collections import defaultdict
import os
import pickle

from PIL import Image
import numpy as np
import torch
import torch.utils.data
import torchvision.transforms as transforms

class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset, part='train', **kwargs):
        super(Dataset, self).__init__()

        with open(dataset, 'rb') as f_ds:
            data = pickle.load(f_ds)
        self.data = data

    def __getitem__(self, index):
        return {
            'x': self.data[index],
        }

    def __len__(self):
        return len(self.data)

def create(*args, **kwargs):
    return Dataset(*args, **kwargs)

if __name__ == '__main__':
    ds_opts = {
        'dataset': 'data/dataset.pkl',
    }
    ds_test = Dataset(**ds_opts)
    print(ds_test[0])
    for i in np.random.permutation(len(ds_test))[:10]:
        ds_test[i]
