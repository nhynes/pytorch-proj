"""Prepares the dataset."""

import argparse
import os
import pickle
import sys


PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(PROJ_ROOT, 'data')
sys.path.append(PROJ_ROOT)

import common


def main():
    # --------------------------------------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument('--out-dir', default='data')
    args = parser.parse_args()
    # --------------------------------------------------------------------------

    out_dir = os.path.join(DATA_DIR, args.out_dir)
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    def _pickle(path_suffix, data):
        with open(os.path.join(out_dir, path_suffix + '.pkl'), 'wb') as f_out:
            pickle.dump(data, f_out)


if __name__ == '__main__':
    main()
