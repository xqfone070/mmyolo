import os
import time

_base_ = [
    'common/yolov8_runtime_schedule_300e.py',
    '../_base_/models/faster-rcnn_yolov8-s_rpn.py',
    'common/coco_detection_custom_640x640_mosaic.py',
]

model_name = 'faster-rcnn_yolov8-s_rpn'
run_name = '%s_%dx%d_%s' % (model_name, _base_.img_scale[0], _base_.img_scale[1], _base_.run_time)
work_dir = os.path.join('work_dirs', _base_.dataset_name, run_name)

batch_size = 16

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
