if __name__ != '__main__':
    exit()

import argparse
import os
import pickle
import traceback

import torch
import torch.nn as nn
import torch.nn.functional as nnf
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

import dataset
import model

#===============================================================================
parser = argparse.ArgumentParser()
# general
parser.add_argument('--seed', default=42, type=int)

# data
parser.add_argument('--dataset', default='data/dataset.pkl')
parser.add_argument('--batch-size', default=64, type=int)
parser.add_argument('--nworkers', default=4, type=int)

# model
parser.add_argument('--emb-dim', default=256, type=int)

# training
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--resume', help='resume epoch')

# output
parser.add_argument('--dispfreq', default=100, type=int)

args = parser.parse_args()
#===============================================================================

for rundir in ['snaps']:
    os.makedirs(f'run/{rundir}', exist_ok=True)

os.mkfifo('run/ctl')
ctl = os.open('run/ctl', flags=os.O_NONBLOCK)

is_resuming = args.resume and os.path.isfile(f'run/snaps/model_{args.resume}.pth')
opts_path = 'run/opts.pkl'
if is_resuming:
    with open(opts_path, 'rb') as f_orig_opts:
        orig_opts = pickle.load(f_orig_opts)
    for k in ['batch_size', 'lr', 'nworkers', 'dispfreq', 'epochs', 'resume']:
        setattr(orig_opts, k, getattr(args, k))
    args = orig_opts

    i = 1
    while os.path.isfile(opts_path):
        opts_path = f'run/opts_{i}.pkl'
        i += 1

for path_arg in ['dataset']:
    setattr(args, path_arg, os.path.abspath(getattr(args, path_arg)))

n_gpu = torch.cuda.device_count()
setattr(args, 'batch_size', int(args.batch_size / n_gpu) * n_gpu)

with open(opts_path, 'wb') as f_opts:
    pickle.dump(args, f_opts)

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

varargs = vars(args)

#===============================================================================

ds_train = dataset.create(**varargs)
ds_val = dataset.create(part='val', **varargs)

loader_opts = {'batch_size': args.batch_size, 'shuffle': True,
               'pin_memory': True, 'num_workers': args.nworkers}
train_loader = torch.utils.data.DataLoader(ds_train, **loader_opts)
val_loader = torch.utils.data.DataLoader(ds_val, **loader_opts)

net = model.create(**varargs)
inputs = {k: Variable(inp.cuda()) for k,inp in net.create_inputs().items()}

optimizer = optim.Adam(net.parameters(), lr=args.lr)

if is_resuming:
    net.load_state_dict(torch.load(f'run/snaps/model_{args.resume}.pth'))
    optimizer.load_state_dict(torch.load(f'run/snaps/optim_state_{args.resume}.pth'))
    print(f'Resuming training from epoch {args.resume}')

if n_gpu > 1:
    net = nn.DataParallel(net)
net = net.cuda()

def do_tasks():
    for cmdline in os.read(ctl, 2**10).decode('utf8').split('\n'):
        cmd_opts = cmdline.split(' ')
        cmd, opts = cmd_opts[0], cmd_opts[1:]
        if cmd == 'val':
            val('usr')
        elif cmd == 'snap':
            snap('usr')
        elif cmd == 'exec':
            patch_path = os.path.expandvars(os.path.expanduser(opts[0]))
            if not os.path.isfile(patch_path):
                continue
            with open(patch_path) as f_patch:
                patch = f_patch.read()
            try:
                exec(compile(patch, patch_path, 'exec'))
            except:
                traceback.print_exc()

def train(i):
    for batch_idx,cpu_inputs in enumerate(train_loader, 1):
        net.train()

        should_disp = args.dispfreq > 0 and \
                (batch_idx % args.dispfreq == 0 or batch_idx == len(train_loader))

        for k,v in inputs.items():
            ct = cpu_inputs[k]
            v.data.resize_(ct.size()).copy_(ct)

        optimizer.zero_grad()

        loss,output = net(**inputs)
        loss.backward()

        optimizer.step()

        if should_disp:
            loss = loss.data[0]
            disp_str = f'[{i}] ({batch_idx}/{len(train_loader)}) | loss: {loss:.5f}'
            print(disp_str)

        do_tasks()

def val(i):
    net.eval()
    val_loss = 0
    for batch_idx,cpu_inputs in enumerate(val_loader, 1):
        for k,v in inputs.items():
            ct = cpu_inputs[k]
            v.data.resize_(ct.size()).copy_(ct)
            v.volatile = True

        loss,output = net(**inputs)
        val_loss += loss.data[0] * len(probs) / args.batch_size

    val_loss = val_loss / len(val_loader)

    disp_str = f'[{i}] (VAL) | loss: {val_loss:.5f}'
    print(disp_str)

    for v in inputs.values():
        v.volatile = False

def state2cpu(state):
    if isinstance(state, dict):
        return type(state)(k: state2cpu(v) for k, v in state)
    elif torch.is_tensor(state):
        return state.cpu()

def snap(i):
    net_mod = net.module if isinstance(net, nn.DataParallel) else net
    torch.save(state2cpu(net_mod.state_dict()), f'run/snaps/model_{i}.pth')
    torch.save(state2cpu(optimizer.state_dict()), f'run/snaps/optim_state_{i}.pth')

try:
    for i in range(1, args.epochs + 1):
        train(i)
        val(i)
        do_tasks()
        snap(i)
        do_tasks()
except KeyboardInterrupt:
    pass
finally:
    os.close(ctl)
    os.unlink('run/ctl')
