import os.path
import argparse

from mmengine.config import Config
from mmengine.runner import load_checkpoint, save_checkpoint
from mmengine.registry import init_default_scope
from mmengine.model.utils import revert_sync_batchnorm
from mmdet.registry import MODELS


def sparsity(model):
    # Return global model sparsity
    a, b = 0, 0
    for p in model.parameters():
        a += p.numel()
        b += (p == 0).sum()
    return b / a


def prune(model, amount=0.5):
    # Prune model to requested global sparsity
    import torch.nn.utils.prune as prune
    import torch.nn as nn
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            prune.l1_unstructured(m, name='weight', amount=amount)  # prune
            prune.remove(m, 'weight')  # make permanent


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMYOLO prunea model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--amount', type=float, default=0.3, help='prune amount')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    config = Config.fromfile(args.config)
    init_default_scope(config.get('default_scope', 'mmdet'))
    model = MODELS.build(config.model)
    # model = revert_sync_batchnorm(model)
    check_point = load_checkpoint(model, args.checkpoint, map_location='cpu')

    print(f'before pruning, global sparsity: {sparsity(model):.3g} ')
    prune(model, args.amount)
    print(f'after pruning, global sparsity: {sparsity(model):.3g} ')

    out_dir = os.path.dirname(args.checkpoint)
    filename = os.path.basename(args.checkpoint)
    file_id = os.path.splitext(filename)[0]
    filename = file_id + '_prune_%.2f' % args.amount + '.pth'
    fullname = os.path.join(out_dir, filename)
    check_point['state_dict'] = model.state_dict()
    save_checkpoint(check_point, fullname)
    print('save checkpoint to %s' % fullname)


main()

