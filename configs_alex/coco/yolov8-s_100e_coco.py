import os
_base_ = ['common/yolov8_runtime_schedule_100e.py',
          'common/coco_detection_custom_640x640_mosaic.py',
          '../_base_/models/yolov8-s-alex.py']


batch_size = 8
# Worker to pre-fetch data for each single GPU during training

model_name = 'yolov8-s-alex'
run_name = '%s_%dx%d_%s' % (model_name, _base_.img_scale[0], _base_.img_scale[1], _base_.run_time)
work_dir = os.path.join('work_dirs', _base_.dataset_name, run_name)

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
