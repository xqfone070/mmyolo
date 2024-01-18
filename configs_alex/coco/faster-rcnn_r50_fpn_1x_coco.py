import os
import time

_base_ = [
    '../_base_/models/faster-rcnn_r50_fpn.py',
    'coco_detection_custom.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]

model_name = 'faster-rcnn_r50_fpn'
dataset_name = 'coco_detection'
time_str = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
run_name = '%s_%dx%d_%s' % (model_name, _base_.img_scale[0], _base_.img_scale[1], time_str)
work_dir = os.path.join('work_dirs', dataset_name, run_name)


wandb_init_kwargs = {'project': dataset_name,
                     'name': run_name}
visualizer = dict(vis_backends=[
    dict(type='LocalVisBackend'),
    dict(type='WandbVisBackend', init_kwargs=wandb_init_kwargs)
])
