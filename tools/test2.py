# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp

from mmdet.engine.hooks.utils import trigger_visualization_hook
from mmengine.config import Config, ConfigDict, DictAction
from mmengine.evaluator import DumpResults
from mmengine.runner import Runner

from mmyolo.registry import RUNNERS
from mmyolo.utils import is_metainfo_lower
import glob

# TODO: support fuse_conv_bn
def parse_args():
    parser = argparse.ArgumentParser(
        description='MMYOLO test (and eval) a model')
    parser.add_argument('work_dir', help='Path of config file, checkpoint file and save dir')
    # find config file automatically in workdir
    # parser.add_argument('config', help='Config file')
    parser.add_argument('--checkpoint', default='best',
                        help='filename or keyword of Checkpoint file, best or epoch number')
    parser.add_argument('--data_root', default=None,
                        help='test data root')
    parser.add_argument(
        '--out',
        type=str,
        help='output result file (must be a .pkl file) in pickle format')
    parser.add_argument(
        '--json-prefix',
        type=str,
        help='the prefix of the output json file without perform evaluation, '
        'which is useful when you want to format the result to a specific '
        'format and submit it to the test server')
    parser.add_argument(
        '--tta',
        action='store_true',
        help='Whether to use test time augmentation')
    parser.add_argument(
        '--show', action='store_true', help='show prediction results')
    parser.add_argument(
        '--deploy',
        action='store_true',
        help='Switch model to deployment mode')
    parser.add_argument(
        '--show-dir',
        help='directory where painted images will be saved. '
        'If specified, it will be automatically saved '
        'to the work_dir/timestamp/show_dir')
    parser.add_argument(
        '--wait-time', type=float, default=2, help='the interval of show (s)')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    # find config file in work dir
    config_file_pattern = os.path.join(args.work_dir, '*.py')
    config_files = glob.glob(config_file_pattern)
    assert (len(config_files) == 1)
    args.config = config_files[0]

    # find checkpoint
    if args.checkpoint.endswith('.pth'):
        args.checkpoint = os.path.join(args.work_dir, args.checkpoint)
    else:
        ck_pattern = os.path.join(args.work_dir, '*%s*.pth' % args.checkpoint)
        ck_files = glob.glob(ck_pattern)
        assert (len(ck_files) == 1)
        args.checkpoint = ck_files[0]
    return args


def main():
    args = parse_args()

    # load config
    cfg = Config.fromfile(args.config)
    # replace the ${key} with the value of cfg.key
    # cfg = replace_cfg_vals(cfg)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # 设置工作目录为原目录下的子目录test下面
    cfg.work_dir = os.path.join(args.work_dir, 'test')
    cfg.load_from = args.checkpoint

    # 替换test_data_root
    if args.data_root is not None:
        img_subdir = 'images'
        ann_subdir = 'annotations'
        set_subdir = 'sets'
        test_ann_file = os.path.join(set_subdir, 'test.txt')
        test_data_cfg = cfg.test_dataloader.dataset
        test_data_cfg.data_root = args.data_root
        test_data_cfg.img_subdir = img_subdir
        test_data_cfg.ann_subdir = ann_subdir
        test_data_cfg.ann_file = test_ann_file
        test_data_cfg.data_prefix.img = img_subdir
        test_data_cfg.data_prefix.sub_data_root = ''

    # 设置单线程加载数据，用于debug
    # cfg.test_dataloader.num_workers = 0
    # cfg.test_dataloader.persistent_workers = False

    # 删除wandb
    cfg.visualizer.vis_backends = [bk for bk in cfg.visualizer.vis_backends if bk.type != 'WandbVisBackend']

    if args.show or args.show_dir:
        cfg = trigger_visualization_hook(cfg, args)

    if args.deploy:
        cfg.custom_hooks.append(dict(type='SwitchToDeployHook'))

    # add `format_only` and `outfile_prefix` into cfg
    if args.json_prefix is not None:
        cfg_json = {
            'test_evaluator.format_only': True,
            'test_evaluator.outfile_prefix': args.json_prefix
        }
        cfg.merge_from_dict(cfg_json)

    # Determine whether the custom metainfo fields are all lowercase
    is_metainfo_lower(cfg)

    if args.tta:
        assert 'tta_model' in cfg, 'Cannot find ``tta_model`` in config.' \
                                   " Can't use tta !"
        assert 'tta_pipeline' in cfg, 'Cannot find ``tta_pipeline`` ' \
                                      "in config. Can't use tta !"

        cfg.model = ConfigDict(**cfg.tta_model, module=cfg.model)
        test_data_cfg = cfg.test_dataloader.dataset
        while 'dataset' in test_data_cfg:
            test_data_cfg = test_data_cfg['dataset']

        # batch_shapes_cfg will force control the size of the output image,
        # it is not compatible with tta.
        if 'batch_shapes_cfg' in test_data_cfg:
            test_data_cfg.batch_shapes_cfg = None
        test_data_cfg.pipeline = cfg.tta_pipeline

    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

    # add `DumpResults` dummy metric
    if args.out is not None:
        assert args.out.endswith(('.pkl', '.pickle')), \
            'The dump file must be a pkl file.'
        runner.test_evaluator.metrics.append(
            DumpResults(out_file_path=args.out))

    # start testing
    runner.test()


if __name__ == '__main__':
    main()
