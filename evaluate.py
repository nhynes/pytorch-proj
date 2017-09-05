import argparse
import pickle

from tqdm import tqdm

import torch
import numpy as np

import common


def main(opts):
    with open('run/opts.pkl', 'rb') as f_opts:
        model_opts = pickle.load(f_opts)
    model_opts.batch_size = opts.batch_size
    model_opts.onelayer = opts.onelayer

    datasets = common.create_datasets(model_opts, partitions=['test'])
    test_loader = common.create_loaders(model_opts, datasets)['test_loader']

    net, inputs = common.create_model(model_opts)
    net.load_state_dict(torch.load(common.MSNAP_PATH.format(opts.epoch)))
    net.eval()

    sz = (len(datasets['test']), model_opts.joint_emb_dim)
    img_embs = np.empty(sz, dtype='float32')
    instr_embs = np.empty(sz, dtype='float32')

    n = 0
    for cpu_inputs in tqdm(test_loader):
        common.copy_inputs(cpu_inputs, inputs, volatile=True)

        outputs = net(**inputs)
        batch_img_embs, batch_instr_embs = outputs[-2:]
        bsl = slice(n, n+len(batch_img_embs))
        img_embs[bsl] = batch_img_embs.data.cpu().numpy()
        instr_embs[bsl] = batch_instr_embs.data.cpu().numpy()
        n += len(batch_img_embs)

    out_suff = f'{opts.epoch}' + (f'_{opts.onelayer}' * bool(opts.onelayer))
    np.save(f'run/instr_embs_{out_suff}.npy', instr_embs)
    np.save(f'run/img_embs_{out_suff}.npy', img_embs)


if __name__ == '__main__':
    # --------------------------------------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument('epoch', type=int)
    parser.add_argument('--batch-size', default=512, type=int)
    parser.add_argument('--onelayer', choices=('l2', 'l4'), default=False)
    opts = parser.parse_args()
    # --------------------------------------------------------------------------

    main(opts)
