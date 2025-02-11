import os
import time
_base_ = '../../configs/yolov8/yolov8_n_syncbn_fast_8xb16-500e_coco.py'
model_name = 'yolov8_swin'

# dataset
dataset_type = 'YOLOv5VOCDataset'
data_root = '/home/alex/data/TPS2000_item_det_train_1028_20231008'  # Root path of data

img_subdir = 'images'
ann_subdir = 'annotations'
# set_subdir = 'sets_b300_r4-1'
set_subdir = 'sets'
train_ann_file = os.path.join(set_subdir, 'trainval.txt')
val_ann_file = os.path.join(set_subdir, 'test.txt')

# classes
class_name = ('item',)  # according to the label information of class_with_id.txt, set the class_name
num_classes = len(class_name)
metainfo = dict(
    classes=class_name,
    palette=[(220, 20, 60)]  # the color of drawing, free to set
)
img_scale = (256, 512)

# weight
load_from = None  # 从给定路径加载模型检查点作为预训练模型。这不会恢复训练。
resume = False  # 是否从 `load_from` 中定义的检查点恢复。 如果 `load_from` 为 None，它将恢复 `work_dir` 中的最新检查点。

dataset_name = os.path.basename(data_root)
time_str = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
run_name = '%s_%dx%d_%s' % (model_name, img_scale[0], img_scale[1], time_str)
work_dir = os.path.join('work_dirs', dataset_name, run_name)

# learning rate
base_lr = 0.01
lr_factor = 0.01
weight_decay = 0.05

# train config
max_epochs = 200
train_batch_size_per_gpu = 16
save_epoch_intervals = 10
train_num_workers = 16  # recommend to use train_num_workers = nGPU x 4


# model
# model = dict(
#     bbox_head=dict(
#         head_module=dict(num_classes=num_classes)
#     ),
#     train_cfg=dict(
#         assigner=dict(num_classes=num_classes)
#     )
# )

deepen_factor = 1.0
widen_factor = 1.0
pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth'  # noqa
model = dict(
    backbone=dict(
        _delete_=True,
        type='mmdet.SwinTransformer',
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(1, 2, 3),
        # Please only add indices that would be used
        # in FPN, otherwise some parameter will not be used
        with_cp=False,
        convert_weights=True,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    neck=dict(deepen_factor=deepen_factor,
              widen_factor=widen_factor,
              in_channels=[192, 384, 768],
              out_channels=[192, 384, 768]),
    bbox_head=dict(
        head_module=dict(
            num_classes=num_classes,
            widen_factor=widen_factor,
            in_channels=[192, 384, 768])
    ),
    train_cfg=dict(
        assigner=dict(num_classes=num_classes)
    )
)


# pipeline
train_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=_base_.file_client_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='mmdet.Resize', scale=img_scale, keep_ratio=False),
    # dict(type='PPYOLOERandomDistort',
    #      hue_cfg=dict(min=-18, max=18, prob=0.0),
    #      saturation_cfg=dict(min=0.5, max=1.5, prob=0.0),
    #      contrast_cfg=dict(min=0.9, max=1.1, prob=0.5),
    #      brightness_cfg=dict(min=-20, max=20, prob=0.5)),
    # dict(
    #     type='YOLOv5RandomAffine',
    #     max_rotate_degree=0.0,
    #     max_shear_degree=0.0,
    #     max_translate_ratio=0.1,
    #     scaling_ratio_range=(0.8, 1.2),
    #     border=(0, 0),
    #     border_val=(0, 0, 0)),
    dict(type='mmdet.RandomFlip', prob=0.5),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'flip',
                   'flip_direction', 'scale_factor'))
]

test_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=_base_.file_client_args),
    dict(type='mmdet.Resize', scale=img_scale, keep_ratio=False),
    dict(type='LoadAnnotations', with_bbox=True, _scope_='mmdet'),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

# dataloader
train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    #num_workers=train_num_workers,
    num_workers=0,
    persistent_workers=False,
    dataset=dict(
        type=dataset_type,
        metainfo=metainfo,
        data_root=data_root,
        img_subdir=img_subdir,
        ann_subdir=ann_subdir,
        ann_file=train_ann_file,
        data_prefix=dict(img=img_subdir, sub_data_root=''),
        filter_cfg=dict(filter_empty_gt=False, min_size=4),
        pipeline=train_pipeline
    )
)

val_dataloader = dict(
    dataset=dict(
        type=dataset_type,
        metainfo=metainfo,
        data_root=data_root,
        img_subdir=img_subdir,
        ann_subdir=ann_subdir,
        ann_file=val_ann_file,
        data_prefix=dict(img=img_subdir, sub_data_root=''),
        pipeline=test_pipeline)
)

test_dataloader = val_dataloader


# evaluator
val_evaluator = dict(
    _delete_=True, type='mmdet.VOCMetric', metric='mAP', eval_mode='area')

test_evaluator = val_evaluator

optim_wrapper = dict(
    optimizer=dict(
        lr=base_lr,
        # weight_decay=weight_decay,
        batch_size_per_gpu=train_batch_size_per_gpu))

train_cfg = dict(
    max_epochs=max_epochs,
    val_begin=1,
    val_interval=1  # the test evaluation is performed  iteratively every val_interval round
)

default_hooks = dict(
    # set how many epochs to save the model, and the maximum number of models to save,`save_best` is also the best model (recommended).
    checkpoint=dict(
        type='CheckpointHook',
        interval=save_epoch_intervals,
        max_keep_ckpts=10),
    param_scheduler=dict(
        lr_factor=lr_factor,
        max_epochs=max_epochs),
    # logger output interval
    logger=dict(type='LoggerHook', interval=10))


custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0001,
        update_buffers=True,
        strict_load=False,
        priority=49),
]


wandb_init_kwargs = {'project': dataset_name,
                     'name': run_name}
visualizer = dict(vis_backends=[
    dict(type='LocalVisBackend'),
    dict(type='WandbVisBackend', init_kwargs=wandb_init_kwargs)
])
