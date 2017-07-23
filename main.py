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

if __name__ != '__main__':
    exit()

# =============================================================================
parser = argparse.ArgumentParser()
# general
parser.add_argument('--seed', default=42, type=int)
parser.add_argument('--debug', action='store_true')

# data
parser.add_argument('--dataset', default='data/dataset.pkl')
parser.add_argument('--nworkers', default=4, type=int)

# model
parser.add_argument('--emb-dim', default=256, type=int)
parser.add_argument('--batch-size', default=64, type=int)

# training
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--resume', help='resume epoch', type=int)

# output
parser.add_argument('--dispfreq', default=100, type=int)

args = parser.parse_args()

for path_arg in ['dataset']:
    setattr(args, path_arg, os.path.abspath(getattr(args, path_arg)))
# =============================================================================
for rundir in ['snaps']:
    os.makedirs(f'run/{rundir}', exist_ok=True)

f_ctl = 'run/ctl'
if not os.path.exists(f_ctl):
    os.mkfifo(f_ctl)
ctl = os.open(f_ctl, flags=os.O_NONBLOCK)

msnap_path = 'run/snaps/model_%s.pth'
osnap_path = f'run/snaps/optim_state_%s.pth'
is_resuming = args.resume and os.path.isfile(msnap_path % args.resume)
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

n_gpu = torch.cuda.device_count()
setattr(args, 'batch_size', int(args.batch_size / n_gpu) * n_gpu)

with open(opts_path, 'wb') as f_opts:
    pickle.dump(args, f_opts)

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

varargs = vars(args)
# =============================================================================

ds_val = dataset.create(part='val', **varargs)
ds_train = dataset.create(part='train', **varargs) if not args.debug else ds_val

loader_opts = {'batch_size': args.batch_size, 'shuffle': True,
               'pin_memory': True, 'num_workers': args.nworkers}
train_loader = torch.utils.data.DataLoader(ds_train, **loader_opts)
val_loader = torch.utils.data.DataLoader(ds_val, **loader_opts)

net = model.create(**varargs)
inputs = {k: Variable(inp.cuda()) for k, inp in net.create_inputs().items()}

optimizer = optim.Adam(net.parameters(), lr=args.lr)

if is_resuming:
    net.load_state_dict(torch.load(msnap_path % args.resume))
    optimizer.load_state_dict(torch.load(osnap_path % args.resume))
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


def train(e):
    nb = len(train_loader)
    for i, cpu_inputs in enumerate(train_loader, 1):
        net.train()

        should_disp = args.dispfreq > 0 and (i % args.dispfreq == 0 or i == nb)

        for k, v in inputs.items():
            if k not in cpu_inputs:
                continue
            ct = cpu_inputs[k]
            v.data.resize_(ct.size()).copy_(ct)

        optimizer.zero_grad()

        loss = net(**inputs)
        loss = loss.mean()  # multi-gpu
        loss.backward()

        optimizer.step()

        if should_disp:
            loss = loss.data[0]
            disp_str = f'[{e}] ({i}/{nb}) | loss: {loss:.5f}'
            print(disp_str)
            with open('run/log.txt', 'a') as f_log:
                print(disp_str, file=f_log, flush=True)

        do_tasks()


def val(e):
    net.eval()
    val_loss = 0
    for i, cpu_inputs in enumerate(val_loader, 1):
        for k, v in inputs.items():
            if k not in cpu_inputs:
                continue
            ct = cpu_inputs[k]
            v.data.resize_(ct.size()).copy_(ct)
            v.volatile = True

        loss = net(**inputs)
        val_loss += loss.mean().data[0]

    val_loss = val_loss / len(val_loader)

    disp_str = f'[{e}] (VAL) | loss: {val_loss:.5f}'
    print(disp_str)
    with open('run/log.txt', 'a') as f_log:
        print(disp_str, file=f_log, flush=True)

    for v in inputs.values():
        v.volatile = False


def state2cpu(state):
    if isinstance(state, dict):
        return type(state)({k: state2cpu(v) for k, v in state.items()})
    elif torch.is_tensor(state):
        return state.cpu()


def snap(e):
    net_mod = net.module if isinstance(net, nn.DataParallel) else net
    torch.save(state2cpu(net_mod.state_dict()), msnap_path % e)
    torch.save(optimizer.state_dict(), osnap_path % e)

try:
    for e in range(args.resume + 1 if args.resume else 1, args.epochs + 1):
        train(e)
        val(e)
        do_tasks()
        snap(e)
        do_tasks()
except KeyboardInterrupt:
    pass
finally:
    os.close(ctl)
    os.unlink('run/ctl')
