_base_ = 'default_runtime_custom.py'

batch_size = 16
max_epochs = 300
interval = 10
base_lr = 0.01
lr_factor = 0.01
weight_decay = 0.0005

param_scheduler = None
optim_wrapper = dict(
    type='OptimWrapper',
    clip_grad=dict(max_norm=10.0),
    optimizer=dict(
        type='SGD',
        lr=base_lr,
        momentum=0.937,
        weight_decay=weight_decay,
        nesterov=True,
        batch_size_per_gpu=batch_size),
    constructor='mmyolo.YOLOv5OptimizerConstructor')

default_hooks = dict(
    checkpoint=dict(interval=interval),
    param_scheduler=dict(
        type='mmyolo.YOLOv5ParamSchedulerHook',
        scheduler_type='linear',
        lr_factor=lr_factor,
        max_epochs=max_epochs)
)


custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0001,
        update_buffers=True,
        strict_load=False,
        priority=49)
]

# train/val/test cfg
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=interval)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
