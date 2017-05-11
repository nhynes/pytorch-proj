import argparse
import os
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as nnf
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

import dataset
import model

if __name__ != '__main__':
    exit()

#===============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('model', help='epoch')
parser.add_argument('--batch-size', default=1024, type=int)
args = parser.parse_args()
#===============================================================================

with open('run/opts.pkl', 'rb') as f_opts:
    orig_args = pickle.load(f_opts)
    for k, v in vars(args).items():
        setattr(orig_args, k, v)
args = orig_args

n_gpu = torch.cuda.device_count()
setattr(args, 'batch_size', int(args.batch_size / n_gpu) * n_gpu)

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

varargs = vars(args)

#===============================================================================

ds_test = dataset.create(part='test', **varargs)

loader_opts = {'batch_size': args.batch_size, 'shuffle': True,
               'pin_memory': True, 'num_workers': args.nworkers}
test_loader = torch.utils.data.DataLoader(ds_test, **loader_opts)

net = model.create(**varargs)
net.load_state_dict(torch.load(f'run/snaps/model_{args.model}.pth'))

inputs = {k: Variable(inp.cuda()) for k, inp in net.create_inputs().items()}

if n_gpu > 1:
    net = nn.DataParallel(net)
net = net.cuda()

net.eval()
for batch_idx, cpu_inputs in enumerate(test_loader, 1):
    for k, v in inputs.items():
        ct = cpu_inputs[k]
        v.data.resize_(ct.size()).copy_(ct)
        v.volatile = True

    # evaluate
