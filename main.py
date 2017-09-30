"""Main training script for the model."""

from collections import defaultdict
from contextlib import contextmanager
import argparse
import os
import pickle
import traceback

import torch
import torch.nn as nn

import common


LOSS_FMT = '{}={:.3f}'
TRAIN_FMT = '[{:d}] ({:d}/{:d}) | loss: {}'
VAL_FMT = '[{:d}] (VAL) | loss: {}'


def main(opts):
    """Trains a model."""
    torch.manual_seed(opts.seed)
    torch.cuda.manual_seed(opts.seed)

    loaders = common.create_loaders(opts, common.create_datasets(opts))
    with _create_runners(opts, **loaders) as (train, val, snap):
        start_epoch = opts.resume_epoch + 1 if opts.resume_epoch else 1
        for epoch in range(start_epoch, opts.epochs + 1):
            train(epoch)
            snap(epoch)
            val(epoch)


@contextmanager
def _create_runners(opts, train_loader, val_loader):
    if (opts.resume_epoch and
            os.path.isfile(common.MSNAP_PATH.format(opts.resume_epoch))):
        print(f'Resuming training from epoch {opts.resume_epoch}')
        net, inputs, optimizer = common.resume(opts)
    else:
        net, inputs = common.create_model(opts)
        optimizer = common.create_optimizer(opts, net)

    n_train_batches = len(train_loader)
    n_val_batches = len(val_loader)
    n_val = len(val_loader.dataset)

    dispfreq = n_train_batches // opts.dispfreq

    f_log = open('run/log.txt', 'a')

    if not os.path.exists(common.CTL_PATH):
        os.mkfifo(common.CTL_PATH)
    f_ctl = os.open(common.CTL_PATH, flags=os.O_NONBLOCK)

    with open(common.OPTS_PATH, 'wb') as f_opts:
        pickle.dump(opts, f_opts)

    def _train(epoch):
        for i, cpu_inputs in enumerate(train_loader, 1):
            net.train()
            common.copy_inputs(cpu_inputs, inputs)

            optimizer.zero_grad()

            outputs = net(**inputs)
            losses = outputs['losses']
            sum(map(torch.mean, losses.values()).backward()

            optimizer.step()

            if dispfreq > 0 and (i % dispfreq == 0 or i == n_train):
                loss_str = ' '.join(
                    LOSS_FMT.format(loss_name, loss_val.data[0])
                    for loss_name, loss_val in losses.items())
                disp_str = TRAIN_FMT.format(epoch, i, n_train, loss_str)
                print(disp_str)
                print(disp_str, file=f_log, flush=True)

            _do_tasks()

    def _val(epoch):
        val_loss = defaultdict(int)
        for i, cpu_inputs in enumerate(val_loader, 1):
            net.eval()
            common.copy_inputs(cpu_inputs, inputs, volatile=True)

            outputs = net(**inputs)
            losses = outputs['losses']
            for loss_name, loss_val in losses.items():
                val_loss[loss_name] += loss_val.mean().data[0]

            _do_tasks()

        for loss_name in val_loss:
            val_loss[loss_name] /= n_val_batches

        loss_str = ' '.join(LOSS_FMT.format(*loss) for loss in val_loss.items())
        disp_str = VAL_FMT.format(epoch, loss_str)
        print(disp_str)
        print(disp_str, file=f_log, flush=True)

    def _snap(epoch):
        net_mod = net.module if isinstance(net, nn.DataParallel) else net
        torch.save(net_mod.state_dict(), common.MSNAP_PATH.format(epoch))
        torch.save(common.state2cpu(net_mod.state_dict()),
                   common.CPU_MSNAP_PATH.format(epoch))
        torch.save(optimizer.state_dict(), common.OSNAP_PATH.format(epoch))
        _do_tasks()

    def _do_tasks():
        for cmdline in os.read(f_ctl, 2**10).decode('utf8').split('\n'):
            cmd_opts = cmdline.split(' ')
            cmd, opts = cmd_opts[0], cmd_opts[1:]
            if cmd == 'val':
                _val('usr')
            elif cmd == 'snap':
                _snap('usr')
            elif cmd == 'exec':
                patch_path = os.path.expandvars(os.path.expanduser(opts[0]))
                if not os.path.isfile(patch_path):
                    continue
                with open(patch_path) as f_patch:
                    patch = f_patch.read()
                    try:
                        patch_vars = dict(globals()).update({
                            'net': net, 'optimizer': optimizer, 'f_log': f_log})
                        exec(compile(patch, patch_path, 'exec'), patch_vars)  # pylint: disable=exec-used
                    except:  # pylint: disable=bare-except
                        traceback.print_exc()

    yield _train, _val, _snap

    f_log.close()
    os.close(f_ctl)
    os.unlink(common.CTL_PATH)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    n_gpu = torch.cuda.device_count()

    # general
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--debug', action='store_true')

    # data
    parser.add_argument('--dataset', type=os.path.abspath,
                        default='data/dataset')
    parser.add_argument('--nworkers', default=4, type=int)

    # model
    parser.add_argument('--batch-size', default=128,
                        type=lambda n: int(n) // n_gpu * n_gpu)

    # training
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--resume-epoch', type=int)

    # output
    parser.add_argument('--dispfreq', default=10, type=int)

    opts = parser.parse_args()
    opts.n_gpu = n_gpu
    # --------------------------------------------------------------------------

    for rundir in common.RUNDIRS:
        os.makedirs(f'run/{rundir}', exist_ok=True)

    try:
        main(opts)
    except KeyboardInterrupt:
        pass
