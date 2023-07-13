import os
import time
_base_ = '../yolov7/yolov7_l_syncbn_fast_8x16b-300e_coco.py'

# dataset
dataset_type = 'YOLOv5VOCDataset'
data_root = '/home/alex_thz_item_det/data/TPS2000_item_det_train_1025_20230710'  # Root path of data
test_data_root = '/home/alex_thz_item_det/data/test_dataset/TPS2000_item_det_test_1004_20230214_shanghai_hongqiaobei'

img_subdir = 'images'
ann_subdir = 'annotations'
train_ann_file = 'sets_voc/trainval.txt'
val_ann_file = 'sets_voc/test.txt'


class_name = ('item',)  # according to the label information of class_with_id.txt, set the class_name
num_classes = len(class_name)
metainfo = dict(
    classes=class_name,
    palette=[(220, 20, 60)]  # the color of drawing, free to set
)

# img_scale = (640, 640)
# anchors = [[(25, 16), (19, 35), (32, 25)], [(23, 51), (34, 41), (31, 63)], [(49, 58), (43, 96), (79, 99)]]

# img_scale = (256, 512)  # width, height
# anchors = [[(9, 12), (12, 17), (8, 28)], [(9, 40), (14, 27), (12, 45)], [(18, 45), (14, 62), (28, 71)]]

img_scale = (160, 320)  # width, height
anchors = [[(6, 8), (5, 18), (8, 12)], [(6, 26), (10, 22), (8, 32)], [(6, 45), (12, 36), (22, 58)]]

# img_scale = (320, 320)
# anchors = [[(12, 8), (9, 17), (15, 11)], [(11, 24), (18, 18), (15, 29)], [(14, 46), (23, 31), (39, 50)]]

# weight
load_from = None  # 从给定路径加载模型检查点作为预训练模型。这不会恢复训练。
resume = False  # 是否从 `load_from` 中定义的检查点恢复。 如果 `load_from` 为 None，它将恢复 `work_dir` 中的最新检查点。
dataset_name = os.path.basename(data_root)
model_name = '{{fileBasenameNoExtension}}_%dx%d' % (img_scale[0], img_scale[1])
time_str = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
run_name = '%s_%s' % (model_name, time_str)
work_dir = os.path.join('work_dirs', dataset_name, run_name)

# learning rate
base_lr = 0.01
lr_factor = 0.01

max_epochs = 200
train_batch_size_per_gpu = 32
save_epoch_intervals = 2
train_num_workers = 16  # recommend to use train_num_workers = nGPU x 4

# loss_cls_weight = _base_.loss_cls_weight * (num_classes / 80 * 3 / _base_.num_det_layers)
# loss_bbox_weight = _base_.loss_bbox_weight * (3 / _base_.num_det_layers)
# loss_obj_weight = _base_.loss_obj_weight * ((img_scale[0] / 640) ** 2 * 3 / _base_.num_det_layers)
loss_cls_weight = 0.0
loss_bbox_weight = 0.05
loss_obj_weight = 0.7

# model
model = dict(
    bbox_head=dict(
        head_module=dict(num_classes=num_classes),
        prior_generator=dict(base_sizes=anchors),

        # loss_cls is dynamically adjusted based on num_classes, but when num_classes = 1, loss_cls is always 0
        loss_cls=dict(loss_weight=loss_cls_weight),
        loss_bbox=dict(loss_weight=loss_bbox_weight),
        loss_obj=dict(loss_weight=loss_obj_weight)
    )
)

# pipeline
train_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=_base_.file_client_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='mmdet.Resize', scale=img_scale, keep_ratio=False),
    # dict(type='YOLOv5HSVRandomAug',
    #      hue_delta=0.0,
    #      saturation_delta=0.0,
    #      value_delta=0.1),
    dict(type='PPYOLOERandomDistort',
         hue_cfg=dict(min=-18, max=18, prob=0.0),
         saturation_cfg=dict(min=0.5, max=1.5, prob=0.0),
         contrast_cfg=dict(min=0.9, max=1.1, prob=0.5),
         brightness_cfg=dict(min=-20, max=20, prob=0.5)),
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
        filter_cfg=dict(filter_empty_gt=False, min_size=16),
        pipeline=train_pipeline
    )
)

batch_shapes_cfg = dict(img_size=img_scale[0])
val_dataloader = dict(
    dataset=dict(
        type=dataset_type,
        metainfo=metainfo,
        data_root=data_root,
        img_subdir=img_subdir,
        ann_subdir=ann_subdir,
        ann_file=val_ann_file,
        data_prefix=dict(img=img_subdir, sub_data_root=''),
        pipeline=test_pipeline,
        batch_shapes_cfg=batch_shapes_cfg)
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
        pipeline=test_pipeline,
        batch_shapes_cfg=batch_shapes_cfg)
)

# evaluator
val_evaluator = dict(
    _delete_=True, type='mmdet.VOCMetric', metric='mAP', eval_mode='area')

test_evaluator = val_evaluator


optim_wrapper = dict(
    optimizer=dict(
        lr=base_lr,
        batch_size_per_gpu=train_batch_size_per_gpu
    )
)


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
        # save_best=['pascal_voc/AP30', 'pascal_voc/AP50', 'pascal_voc/mAP'],
        # rule='greater'
    ),
    param_scheduler=dict(
        lr_factor=lr_factor,
        max_epochs=max_epochs),
    # logger output interval
    logger=dict(type='LoggerHook', interval=10))

wandb_init_kwargs = {'project': dataset_name,
                     'name': run_name}
visualizer = dict(vis_backends=[
    dict(type='LocalVisBackend'),
    dict(type='WandbVisBackend', init_kwargs=wandb_init_kwargs)
])