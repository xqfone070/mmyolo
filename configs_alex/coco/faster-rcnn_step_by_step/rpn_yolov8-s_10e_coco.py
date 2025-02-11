import os
import time

_base_ = [
    '../common/yolov8_runtime_schedule_10e.py',
    '../../_base_/models/rpn_yolov8-s.py',
    '../common/coco_detection_custom_640x640_mosaic.py',
]

model_name = 'rpn_yolov8-s'
run_name = '%s_%dx%d_%s' % (model_name, _base_.img_scale[0], _base_.img_scale[1], _base_.run_time)
work_dir = os.path.join('work_dirs', _base_.dataset_name, run_name)

load_from = 'work_dirs/coco_detection/yolov8-s-alex_640x640_20240205_104849/best_coco_bbox_mAP_epoch_100.pth'

# 必须设置，否则冻结rpn_head后会报错
find_unused_parameters = True

# freeze backbone and neck, train head
model = dict(
    backbone=dict(frozen_stages=4),
    neck=dict(freeze_all=True),
)


batch_size = 8

val_evaluator = dict(metric='proposal_fast')
test_evaluator = val_evaluator


# wandb
wandb_init_kwargs = dict(
    project=_base_.dataset_name,
    name=run_name,
)
_base_.visualizer.vis_backends[1].init_kwargs = wandb_init_kwargs


# reset batch size
_base_.optim_wrapper.optimizer.batch_size_per_gpu = batch_size

train_dataloader = dict(
    #    num_workers=0,
    #    persistent_workers=False,
    batch_size=batch_size
)
