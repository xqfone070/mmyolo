_base_ = '../yolov8/yolov8_l_syncbn_fast_8xb16-500e_coco.py'


# data related
dataset_type = 'YOLOv5VOCDataset'
data_root = '/home/alex/data/TPS2000_item_det_20230315/'  # Root path of data
test_data_root = '/home/alex/data/test_dataset/TPS2000_item_det_test_1004_20230214_shanghai_hongqiaobei'
img_subdir = 'images'
ann_subdir = 'annotations'
train_ann_file = 'sets_voc/trainval.txt'
val_ann_file = 'sets_voc/test.txt'

img_scale = (160, 320)
class_name = ('item',)  # according to the label information of class_with_id.txt, set the class_name
num_classes = len(class_name)
metainfo = dict(
    classes=class_name,
    palette=[(220, 20, 60)]  # the color of drawing, free to set
)


# weight
load_from = None  # 从给定路径加载模型检查点作为预训练模型。这不会恢复训练。
resume = False  # 是否从 `load_from` 中定义的检查点恢复。 如果 `load_from` 为 None，它将恢复 `work_dir` 中的最新检查点。

# learning rate
base_lr = 0.02
lr_factor = 0.01

# train config
max_epochs = 100
train_batch_size_per_gpu = 32
save_epoch_intervals = 2
train_num_workers = 16  # recommend to use train_num_workers = nGPU x 4

# model
model = dict(
    bbox_head=dict(
        head_module=dict(num_classes=num_classes)
    ),
    train_cfg=dict(
        assigner=dict(num_classes=num_classes)
    )
)

random_affine = dict(
        type='YOLOv5RandomAffine',
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        scaling_ratio_range=(1 - _base_.affine_scale, 1 + _base_.affine_scale),
        max_aspect_ratio=_base_.max_aspect_ratio,
        # img_scale is (width, height)
        border=(-img_scale[0] // 2, -img_scale[1] // 2),
        border_val=(114, 114, 114))

# pipeline
train_pipeline = [
    *_base_.pre_transform,
    dict(
        type='Mosaic',
        img_scale=img_scale,
        pad_val=114.0,
        pre_transform=_base_.pre_transform),
    random_affine,
    *_base_.last_transform
]

train_pipeline_stage2 = [
    *_base_.pre_transform,
    dict(type='YOLOv5KeepRatioResize', scale=img_scale),
    dict(
        type='LetterResize',
        scale=img_scale,
        allow_scale_up=True,
        pad_val=dict(img=114.0)),
    random_affine,
    *_base_.last_transform
]


test_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=_base_.file_client_args),
    dict(type='YOLOv5KeepRatioResize', scale=img_scale),
    dict(
        type='LetterResize',
        scale=img_scale,
        allow_scale_up=False,
        pad_val=dict(img=114)),
    dict(type='LoadAnnotations', with_bbox=True, _scope_='mmdet'),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'pad_param'))
]

# dataloader
train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    dataset=dict(
        type=dataset_type,
        metainfo=metainfo,
        data_root=data_root,
        img_subdir=img_subdir,
        ann_subdir=ann_subdir,
        ann_file=train_ann_file,
        data_prefix=dict(img=img_subdir, sub_data_root=''),
        filter_cfg=dict(filter_empty_gt=False, min_size=8),
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

test_dataloader = dict(
    dataset=dict(
        type=dataset_type,
        metainfo=metainfo,
        data_root=test_data_root,
        img_subdir=img_subdir,
        ann_subdir=ann_subdir,
        ann_file=val_ann_file,
        data_prefix=dict(img=img_subdir, sub_data_root=''),
        pipeline=test_pipeline)
)

# evaluator
val_evaluator = dict(
    _delete_=True, type='mmdet.VOCMetric', metric='mAP', eval_mode='area')

test_evaluator = val_evaluator


optim_wrapper = dict(
    optimizer=dict(
        lr=base_lr,
        batch_size_per_gpu=train_batch_size_per_gpu))


train_cfg = dict(
    max_epochs=max_epochs,
    val_begin=1,
    val_interval=1,  # the test evaluation is performed  iteratively every val_interval round
    dynamic_intervals=[((max_epochs - _base_.close_mosaic_epochs),
                        _base_.val_interval_stage2)]
)

default_hooks = dict(
    # set how many epochs to save the model, and the maximum number of models to save,`save_best` is also the best model (recommended).
    checkpoint=dict(
        type='CheckpointHook',
        interval=save_epoch_intervals,
        max_keep_ckpts=5),
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
    dict(
        type='mmdet.PipelineSwitchHook',
        switch_epoch=max_epochs - _base_.close_mosaic_epochs,
        switch_pipeline=train_pipeline_stage2)
]

visualizer = dict(vis_backends=[dict(type='LocalVisBackend'), dict(type='WandbVisBackend')])
