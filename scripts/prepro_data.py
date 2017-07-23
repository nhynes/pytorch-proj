from collections import defaultdict
from tqdm import tqdm
import argparse
import os
import pickle
import random
import re
import sys

PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_ROOT = os.path.join(PROJ_ROOT, 'data')
sys.path.append(PROJ_ROOT)

if __name__ == '__main__':

  #=============================================================================
  parser = argparse.ArgumentParser()
  parser.add_argument('--seed', type=int, default=42)
  args = parser.parse_args()
  #=============================================================================

    random.seed(args.seed)

    data = {}

    with open(os.path.join(DATA_ROOT, 'dataset.pkl'), 'wb') as f_out:
        pickle.dump(data, f_out)
