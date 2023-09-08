import os.path
import time
_base_ = '../../configs/yolov5/yolov5_s-v61_syncbn_8xb16-300e_coco.py'
model_name = 'yolov5_s'

# dataset
dataset_type = 'YOLOv5VOCDataset'
data_root = '/home/alex/data/TPS2000_item_det_train_1026_20230718'

img_subdir = 'images'
ann_subdir = 'annotations'
# set_subdir = 'sets_voc'
set_subdir = 'sets_b300_r4-1'
train_ann_file = os.path.join(set_subdir, 'trainval.txt')
val_ann_file = os.path.join(set_subdir, 'test.txt')


class_name = ('item', )  # according to the label information of class_with_id.txt, set the class_name
num_classes = len(class_name)
metainfo = dict(
    classes=class_name,
    palette=[(220, 20, 60)]  # the color of drawing, free to set
)

# img_scale = (160, 320)
# anchors = [[(6, 8), (5, 18), (8, 12)], [(6, 26), (9, 21), (8, 31)], [(6, 44), (12, 35), (21, 58)]]

img_scale = (320, 640)
anchors = [[(12, 17), (10, 36), (15, 24)], [(12, 51), (19, 42), (16, 62)], [(12, 88), (23, 71), (40, 115)]]


# weight
load_from = None  # 从给定路径加载模型检查点作为预训练模型。这不会恢复训练。
resume = False  # 是否从 `load_from` 中定义的检查点恢复。 如果 `load_from` 为 None，它将恢复 `work_dir` 中的最新检查点。
dataset_name = os.path.basename(data_root)
time_str = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
run_name = '%s_%dx%d_%s' % (model_name, img_scale[0], img_scale[1], time_str)
work_dir = os.path.join('work_dirs', dataset_name, run_name)

# learning rate
base_lr = 0.01
lr_factor = 0.1

max_epochs = 200
train_batch_size_per_gpu = 64
save_epoch_intervals = 10
train_num_workers = 4  # recommend to use train_num_workers = nGPU x 4

# only on Val
batch_shapes_cfg = dict(img_size=img_scale[0])

loss_cls_weight = 0.5
loss_bbox_weight = 0.05
loss_obj_weight = 1.0
num_det_layers = 3

# model
model = dict(
    bbox_head=dict(
        head_module=dict(num_classes=num_classes),
        prior_generator=dict(base_sizes=anchors),

        loss_cls=dict(
            loss_weight=loss_cls_weight *
                        (num_classes / 80 * 3 / num_det_layers)),
        loss_bbox=dict(loss_weight=loss_bbox_weight * (3 / num_det_layers)),
        loss_obj=dict(
            loss_weight=loss_obj_weight *
                        ((img_scale[0] / 640) ** 2 * 3 / num_det_layers)),
        prior_match_thr=4.0
    )
)

# pipeline
train_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=_base_.file_client_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='mmdet.Resize', scale=img_scale, keep_ratio=False),
    dict(type='PPYOLOERandomDistort',
         hue_cfg=dict(min=-18, max=18, prob=0.0),
         saturation_cfg=dict(min=0.5, max=1.5, prob=0.0),
         contrast_cfg=dict(min=0.7, max=1.3, prob=0.5),
         brightness_cfg=dict(min=-30, max=30, prob=0.5)),
    dict(
        type='YOLOv5RandomAffine',
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        max_translate_ratio=0.1,
        scaling_ratio_range=(0.8, 1.2),
        # img_scale is (width, height)
        border=(0, 0),
        border_val=(0, 0, 0)),
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
        _delete_=True,
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
        pipeline=test_pipeline,
        batch_shapes_cfg=batch_shapes_cfg)
)

test_dataloader = val_dataloader

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
    # set how many epochs to save the model, and the maximum number of models to save,
    # `save_best` is also the best model (recommended).
    checkpoint=dict(
        type='CheckpointHook',
        interval=save_epoch_intervals,
        max_keep_ckpts=10
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