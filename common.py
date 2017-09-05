"""Utility functions for training neural networks."""

import os
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable


import model
import dataset


MSNAP_PATH = 'run/snaps/model_{:d}.pth'
CPU_MSNAP_PATH = 'run/snaps/model_{:d}_cpu.pth'
OSNAP_PATH = 'run/snaps/optim_state_{:d}.pth'
OPTS_PATH = 'run/opts.pkl'
CTL_PATH = 'run/ctl'
OVERWRITE_OPTS = {'batch_size', 'lr', 'nworkers',
                  'dispfreq', 'epochs', 'resume_epoch'}
RUNDIRS = ['snaps']


def resume(opts):
    """Returns a model, its inputs, and optimizer loaded from a given epoch."""
    opts_path = OPTS_PATH
    with open(opts_path, 'rb') as f_opts:
        init_opts = pickle.load(f_opts)

    for opt_name, opt_val in vars(init_opts).items():
        if opt_name not in OVERWRITE_OPTS:
            setattr(opts, opt_name, opt_val)

    i = 1
    while os.path.isfile(opts_path):
        opts_path = f'run/opts_{i}.pkl'
        i += 1
    with open(opts_path, 'wb') as f_new_opts:
        pickle.dump(opts, f_new_opts)

    net, inputs = create_model(opts)
    net.load_state_dict(torch.load(MSNAP_PATH.format(opts.resume_epoch)))

    optimizer = create_optimizer(opts, net)
    optimizer.load_state_dict(torch.load(OSNAP_PATH.format(opts.resume_epoch)))

    return net, inputs, optimizer


def create_datasets(opts, partitions=('train', 'val')):
    """Returns a mapping from the provided partitions to `Dataset`s."""
    part_datasets = {part: dataset.create(part=part, **vars(opts))
                     for part in partitions}
    opts.n_topics = next(iter(part_datasets.values())).n_topics
    return part_datasets


def create_loaders(opts, datasets):
    """Returns loaders for a mapping from partition to `Dataset`."""
    loader_opts = {'batch_size': opts.batch_size, 'pin_memory': True,
                   'num_workers': opts.nworkers}
    return {
        f'{part}_loader':
        torch.utils.data.DataLoader(ds, shuffle=part == 'train', **loader_opts)
        for part, ds in datasets.items()}


def create_model(opts):
    """Returns a fresh instance of a model and its inputs."""
    net = model.create(**vars(opts))
    inputs = {k: Variable(inp.cuda()) for k, inp in net.create_inputs().items()}
    inputs['epoch'] = 0
    if opts.n_gpu > 1:
        net = nn.DataParallel(net)
    net = net.cuda()
    return net, inputs


def create_optimizer(opts, model):
    """Returns an optimizer for the provided model."""
    optimizer = optim.Adam(model.parameters(), lr=opts.lr)
    return optimizer


def state2cpu(state):
    """Moves `Tensor`s in state dict to the CPU."""
    if isinstance(state, dict):
        return type(state)({k: state2cpu(v) for k, v in state.items()})
    elif torch.is_tensor(state):
        return state.cpu()


def copy_inputs(cpu_inputs, inputs, volatile=False):
    """Copies Tensors into Variables."""
    for input_name, inp in inputs.items():
        if input_name not in cpu_inputs:
            continue
        cpu_tensor = cpu_inputs[input_name]
        inp.data.resize_(cpu_tensor.size()).copy_(cpu_tensor)
        if isinstance(inp, Variable):
            inp.volatile = volatile
