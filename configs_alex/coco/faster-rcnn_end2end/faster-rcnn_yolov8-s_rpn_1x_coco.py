import os
import time


_base_ = [
    'common/default_runtime_custom.py',
    '../_base_/models/faster-rcnn_yolov8_bone_neck_head.py',
    'mmdet::_base_/schedules/schedule_1x.py',
    'common/coco_detection_custom_640x640_mosaic.py',
]


batch_size = 16

model_name = 'faster-rcnn_yolov8-s_bone_neck_head'
run_name = '%s_%dx%d_%s' % (model_name, _base_.img_scale[0], _base_.img_scale[1], _base_.run_time)
work_dir = os.path.join('work_dirs', _base_.dataset_name, run_name)

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
