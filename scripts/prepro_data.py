import argparse
import os
import pickle


PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_ROOT = os.path.join(PROJ_ROOT, 'data')
sys.path.append(PROJ_ROOT)


def main(args):
    data = {}

    with open(os.path.join(DATA_ROOT, 'dataset.pkl'), 'wb') as f_out:
        pickle.dump(data, f_out)


if __name__ == '__main__':
    # ----------------------------------------------------------------------------
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    # ----------------------------------------------------------------------------

    main(args)

