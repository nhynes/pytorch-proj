"""A Dataset that loads the dataset."""

import os
import pickle

from PIL import Image
import numpy as np

from torchvision import transforms
import torch
import torch.utils.data


EPOCH_SAMPLES = 2_000_000

PAD, UNK, EOS = 0, 1, 2
EOS_TOKS = {'.', '!'}


def _load_txt(path_txt):
    with open(path_txt) as f_txt:
        return [line.rstrip() for line in f_txt]


def _unpickle(path_pkl):
    with open(path_pkl, 'rb') as f_pkl:
        return pickle.load(f_pkl)


class Dataset(torch.utils.data.Dataset):
    """Loads the data."""

    def __init__(self, dataset, images_dir, vocab, vocab_size, max_seqlen,
                 part, **unused_kwargs):
        super(Dataset, self).__init__()

        self._max_seqlen = max_seqlen
        self._part = part
        self._sample_counter = 0

        if part == 'train':
            extra_txforms = [
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
            ]
        else:
            extra_txforms = [transforms.CenterCrop(224)]
        self.transform = transforms.Compose([
            transforms.Scale(256),
            *extra_txforms,
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        vocab = _load_txt(vocab)
        self._vocab = ['PAD', 'UNK', '</s>'] + vocab[:vocab_size-3]
        self._w2i = {w: i for i, w in enumerate(self._vocab)}

        self._samples = _unpickle(f'{dataset}/{part}.pkl')

    def _load_image(self, image_path):
        try:
            return self.transform(Image.open(image_path).convert('RGB')), True
        except IOError:
            with open('data/blacklist.txt', 'a') as f_bl:
                print(image_path, file=f_bl)
            return torch.zeros(3, 224, 224), False

    def _tokenize(self, text):
        toks = []
        for sent in text:
            for tok in sent:
                if len(toks) == self._max_seqlen:
                    break
                toks.append(self._w2i.get(tok, UNK))
            if len(toks) == self._max_seqlen:
                break
            if tok not in EOS_TOKS:
                toks.append(EOS)
        return toks

    def __getitem__(self, index):
        if self._part == 'train':
            n_folds = len(self._samples) // EPOCH_SAMPLES
            if n_folds > 0:
                fold = (self._sample_counter // EPOCH_SAMPLES) % n_folds
                index = (index * n_folds) + fold
            self._sample_counter += 1

        text, image_path = self._samples[index]

        toks = torch.LongTensor(self._max_seqlen).zero_()
        image, has_image = self._load_image(image_path)

        toks_list = self._tokenize(text)
        tok_offset = max(0, int((self._max_seqlen - len(toks_list)) / 2) - 1)
        for i, tok in enumerate(toks_list, tok_offset):
            toks[i] = tok

        return {
            'images': image,
            'texts': toks,
        }

    def __len__(self):
        return min(len(self._samples), EPOCH_SAMPLES)

    def _decode_text(self, text):
        toks = [self._vocab[tok] for tok in text if tok > 0]
        return ' '.join(toks)


def create(*args, **kwargs):
    """Returns a Dataset."""
    return Dataset(*args, **kwargs)


if __name__ == '__main__':
    ds_opts = {
        'data': 'data/data',
        'part': 'test',
    }

    ds_test = Dataset(**ds_opts)
    datum = ds_test[0]

    # for i in np.random.permutation(len(ds_test))[:1000]:
    #     datum = ds_test[i]
