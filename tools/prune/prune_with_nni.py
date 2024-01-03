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


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMYOLO prune a model')
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

    from nni.compression.pytorch.pruning import L1NormPruner
    config_list = [{
        'sparsity_per_layer': args.amount,
        'op_types': ['Conv2d']
    }]

    pruner = L1NormPruner(model, config_list)

    print(model)
    # compress the model and generate the masks
    _, masks = pruner.compress()
    # show the masks sparsity
    for name, mask in masks.items():
        print(name, ' sparsity : ', '{:.2}'.format(mask['weight'].sum() / mask['weight'].numel()))

    # need to unwrap the model, if the model is wrapped before speedup
    pruner._unwrap_model()

    # speedup the model, for more information about speedup, please refer :doc:`pruning_speedup`.
    from nni.compression.pytorch.speedup import ModelSpeedup
    import torch
    ModelSpeedup(model, torch.rand(3, 3, 640, 320), masks).speedup_model()
    print(model)

    out_dir = os.path.dirname(args.checkpoint)
    filename = os.path.basename(args.checkpoint)
    file_id = os.path.splitext(filename)[0]
    filename = file_id + '_prune_%.2f' % args.amount + '.pth'
    fullname = os.path.join(out_dir, filename)
    check_point['state_dict'] = model.state_dict()
    save_checkpoint(check_point, fullname)
    print('save checkpoint to %s' % fullname)


def test_nni():
    from torchvision import models
    import torch
    model = models.resnet50(pretrained=True)

    from nni.compression.pytorch.pruning import L1NormPruner
    config_list = [{
        'sparsity_per_layer': 0.5,
        'op_types': ['Conv2d']
    }]

    pruner = L1NormPruner(model, config_list)

    print(model)
    # compress the model and generate the masks
    _, masks = pruner.compress()
    # show the masks sparsity
    for name, mask in masks.items():
        print(name, ' sparsity : ', '{:.2}'.format(mask['weight'].sum() / mask['weight'].numel()))

    # need to unwrap the model, if the model is wrapped before speedup
    pruner._unwrap_model()

    # speedup the model, for more information about speedup, please refer :doc:`pruning_speedup`.
    from nni.compression.pytorch.speedup import ModelSpeedup
    import torch
    ModelSpeedup(model, torch.rand(1, 3, 224, 224), masks).speedup_model()
    print(model)
    torch.save(model, '/home/alex/resnet50_pruned.pth')


def test_load_pruned_model():
    model_file = '/home/alex/vgg16_pruned.pth'
    # from torchvision import models
    import torch
    # model = models.resnet50(pretrained=False)
    model = torch.load(model_file)
    print(model)


# main()
#test_nni()
test_load_pruned_model()


