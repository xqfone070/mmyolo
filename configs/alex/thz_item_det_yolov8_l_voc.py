_base_ = '../yolov8/yolov8_l_syncbn_fast_8xb16-500e_coco.py'


# data related
dataset_type = 'YOLOv5VOCDataset'
data_root = '/home/alex/data/TPS2000_item_det_20230315/'  # Root path of data
test_data_root = '/home/alex/data/test_dataset/TPS2000_item_det_test_1004_20230214_shanghai_hongqiaobei'
# data_root = r'D:\01.data\05.train_data\TPS2000_item_det_train_1019_20230530/'
img_subdir = 'images'
ann_subdir = 'annotations'
train_ann_file = 'sets_voc/trainval.txt'
val_ann_file = 'sets_voc/test.txt'

img_scale = (256, 512)
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

# pipeline
train_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=_base_.file_client_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='mmdet.Resize', scale=img_scale, keep_ratio=False),
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
    _delete_=True, type='mmdet.VOCMetric', iou_thrs=[0.3, 0.5], metric='mAP', eval_mode='area')

test_evaluator = val_evaluator


optim_wrapper = dict(
    optimizer=dict(
        lr=base_lr,
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
        max_keep_ckpts=5,
        save_best=['pascal_voc/AP30', 'pascal_voc/AP50', 'pascal_voc/mAP'],
        rule='greater'),
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

visualizer = dict(vis_backends=[dict(type='LocalVisBackend'), dict(type='WandbVisBackend')])
