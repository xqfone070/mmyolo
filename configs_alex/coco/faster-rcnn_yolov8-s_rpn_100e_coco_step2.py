import os
import time

_base_ = [
    'common/default_runtime_custom.py',
    '../_base_/models/faster-rcnn_yolov8-s_rpn.py',
    '../_base_/schedules/schedule_cosine_100e.py',
    'common/coco_detection_custom_640x640_mosaic.py',
]

# 注意确认faster_rcnn与load_from的rpn网络的预处理data_preprocessor参数是一致的
load_from = 'work_dirs/coco_detection/rpn_yolov8-s_640x640_20240201_165901/best_coco_AR@100_epoch_72.pth'

batch_size = 16

model_name = 'faster-rcnn_yolov8-s_rpn'
run_name = '%s_%dx%d_%s' % (model_name, _base_.img_scale[0], _base_.img_scale[1], _base_.run_time)
work_dir = os.path.join('work_dirs', _base_.dataset_name, run_name)


# 必须设置，否则冻结rpn_head后会报错
find_unused_parameters = True

# freeze backbone and neck
model = dict(
    backbone=dict(frozen_stages=4),
    neck=dict(freeze_all=True),
    rpn_head=dict(
        freeze_all=True,
        loss_cls=dict(loss_weight=0.0),
        loss_bbox=dict(loss_weight=0.0),
        loss_dfl=dict(loss_weight=0.0),
    ),
)

# yolo系列与faster-rcnn的optimizer的constructor是不同的，从这个角度看也是要分两步训练的
# freeze rpn_head
optim_wrapper = dict(
    # paramwise_cfg=dict(
    #     custom_keys={
    #         'rpn_head': dict(lr_mult=0.0)
    #     }),
    # faster-rcnn use DefaultOptimWrapperConstructor
    constructor='DefaultOptimWrapperConstructor'
)

# wandb
wandb_init_kwargs = dict(
    project=_base_.dataset_name,
    name=run_name,
)
_base_.visualizer.vis_backends[1].init_kwargs = wandb_init_kwargs

train_dataloader = dict(
    #    num_workers=0,
    #    persistent_workers=False,
    batch_size=batch_size
)
