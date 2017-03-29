import argparse
import os
import re
import sys

import numpy as np
import matplotlib.pyplot as plt

PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_ROOT = os.path.join(PROJ_ROOT, 'data')
EXP_ROOT = os.path.join(PROJ_ROOT, 'experiments')
sys.path.append(PROJ_ROOT)

STATS_RE = re.compile(r'\[([1-9][0-9]*)\] \((\d+)/(\d+)\).*\|.*loss: (\d+\.\d+)')

def read_stats(exp_name):
    log_path = os.path.join(EXP_ROOT, exp_name, 'run', 'log.txt')
    ts = []
    losses = []
    epoch_ts = []
    with open(log_path) as f_stats:
        for l in f_stats:
            m = STATS_RE.match(l.rstrip())
            if not m:
                continue
            epoch, itr, itr_per_epoch, loss = m.groups()
            t = (int(epoch) - 1)*int(itr_per_epoch) + int(itr)
            if itr == itr_per_epoch:
                epoch_ts.append(t)
            if len(losses) and float(loss) > 3*losses[-1]:
                continue # wrong scaling
            ts.append(t)
            losses.append(float(loss))

    return ts, losses, epoch_ts

if __name__ != '__main__':
    exit()

#======================================================================================
parser = argparse.ArgumentParser()
parser.add_argument('exp_name', nargs='+')
args = parser.parse_args()
#======================================================================================

plt.figure()

min_loss = float('inf')
max_loss = 0
epoch_ts = []
for exp_name in args.exp_name:
    ts, losses, ets = read_stats(exp_name)
    if len(ets) > len(epoch_ts):
        epoch_ts = ets
    min_loss = min(min_loss, *losses)
    max_loss = max(max_loss, *losses)
    plt.plot(ts, losses, label=exp_name)

plt.vlines(epoch_ts, ymin=min_loss, ymax=max_loss, linestyles='dashed')

plt.title('train loss')
plt.xlabel('iter')
plt.ylabel('loss')
plt.legend()

plt.show()
plt.close()
