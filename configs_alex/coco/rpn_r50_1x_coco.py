import os

_base_ = [
    'common/default_runtime_custom.py',
    'mmdet::_base_/models/rpn_r50_fpn.py',
    'mmdet::_base_/schedules/schedule_1x.py',
    'common/coco_detection_custom_640x640_mosaic.py',
]


model_name = 'rpn_r50'
run_name = '%s_%dx%d_%s' % (model_name, _base_.img_scale[0], _base_.img_scale[1], _base_.run_time)
work_dir = os.path.join('work_dirs', _base_.dataset_name, run_name)

val_evaluator = dict(metric='proposal_fast')
test_evaluator = val_evaluator

# wandb
wandb_init_kwargs = dict(
    project=_base_.dataset_name,
    name=run_name,
)

_base_.visualizer.vis_backends[1].init_kwargs = wandb_init_kwargs


