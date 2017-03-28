if __name__ != '__main__':
    exit()

import argparse
import os
import pickle
import signal
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
parser.add_argument('--dataset', default='data/ingredients.pkl')
parser.add_argument('--train-frac', default=0.9, type=float)
parser.add_argument('--vocab-size', default=5000, type=int)
parser.add_argument('--name-vocab-size', default=4000, type=int)
parser.add_argument('--batch-size', default=64, type=int)
parser.add_argument('--nworkers', default=4, type=int)
parser.add_argument('--max-seqlen', default=18, type=int)

# model
parser.add_argument('--word-emb-dim', default=64, type=int)
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

resuming = args.resume and os.path.isfile(f'run/snaps/model_{args.resume}.pth')
opts_path = 'run/opts.pkl'
if resuming:
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
ds_val = dataset.create(val=True, **varargs)

loader_opts = {'batch_size': args.batch_size, 'shuffle': True,
               'pin_memory': True, 'num_workers': args.nworkers}
train_loader = torch.utils.data.DataLoader(ds_train, **loader_opts)
val_loader = torch.utils.data.DataLoader(ds_val, **loader_opts)

net = model.create(**varargs)
inputs = {k: Variable(inp.cuda()) for k,inp in net.create_inputs().items()}
if n_gpu > 1:
    net = nn.DataParallel(net)
net = net.cuda()

optimizer = optim.Adam(net.parameters(), lr=args.lr)

if resuming:
    net.load_state_dict(torch.load(f'run/snaps/model_{args.resume}.pth'))
    optimizer.load_state_dict(torch.load(f'run/snaps/optim_state_{args.resume}.pth'))
    print(f'Resuming training from epoch {args.resume}')

tasks = []
def do_tasks():
    while tasks:
        task = tasks.pop(0)
        task[0](*task[1:])

def train(i):
    for batch_idx,cpu_inputs in enumerate(train_loader, 1):
        net.train()

        should_disp = args.dispfreq > 0 and \
                (batch_idx % args.dispfreq == 0 or batch_idx == len(train_loader))

        for k,v in inputs.items():
            ct = cpu_inputs[k]
            v.data.resize_(ct.size()).copy_(ct)

        optimizer.zero_grad()

        loss,probs = net(**inputs)
        confs = inputs['confidences'].unsqueeze(1)
        probs.register_hook(lambda g: g * confs.expand(g.size()))
        loss.backward()

        optimizer.step()

        if should_disp:
            loss = loss.data[0]
            disp_str = f'[{i}] ({batch_idx}/{len(train_loader)}) | loss: {loss:.5f}'
            print(disp_str)
            with open('run/log.txt', 'a') as f_log:
                print(disp_str, file=f_log)

        do_tasks()

def val(i):
    net.eval()
    val_loss = 0
    c1 = c5 = n_tgt = 0
    for batch_idx,cpu_inputs in enumerate(train_loader, 1):
        for k,v in inputs.items():
            ct = cpu_inputs[k]
            v.data.resize_(ct.size()).copy_(ct)
            v.volatile = True

        loss,probs = net(**inputs)
        val_loss += loss.data[0] * len(probs) / args.batch_size
        preds = probs.topk(5)[1]
        names = inputs['names']
        c1 += (preds[:,0] == names).sum().data[0]
        c5 += (preds == names.unsqueeze(1).expand(preds.size())).sum().data[0]
        n_tgt += names.numel()

    a1 = c1 / n_tgt * 100
    a5 = c5 / n_tgt * 100
    val_loss = val_loss / len(train_loader)

    disp_str = f'[{i}] (VAL) | loss: {val_loss:.5f}   acc@1: {a1:.1f}   acc@5: {a5:.1f}'
    print(disp_str)
    with open('run/log.txt', 'a') as f_log:
        print(disp_str, file=f_log)

    for v in inputs.values():
        v.volatile = False

def snap(i):
    torch.save(net.state_dict(), f'run/snaps/model_{i}.pth')
    torch.save(optimizer.state_dict(), f'run/snaps/optim_state_{i}.pth')

def dofile():
    patch_path = 'run/patch.py'
    if not os.path.isfile(patch_path):
        return
    with open(patch_path) as f_patch:
        try:
            exec(f_patch.read())
        except:
            traceback.print_exc()

signal.signal(signal.SIGUSR1, lambda signum,stack: tasks.append((snap, 'usr')))
signal.signal(signal.SIGUSR2, lambda signum,stack: tasks.append((val, 'usr')))
signal.signal(signal.SIGTRAP, lambda signum,stack: tasks.append((dofile,)))

for i in range(1, args.epochs + 1):
    train(i)
    val(i)
    do_tasks()
    snap(i)
    do_tasks()
