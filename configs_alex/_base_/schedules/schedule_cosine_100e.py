_base_ = 'mmdet::_base_/schedules/schedule_1x.py'

warmup_epochs = 5
max_epochs = 100
num_last_epochs = 5

# training schedule for 1x
train_cfg = dict(max_epochs=max_epochs, val_interval=1)

base_lr = _base_.optim_wrapper.optimizer.lr
lr_factor = 0.01  # Learning rate scaling factor

# learning rate
# refer to yolox
param_scheduler = [
    dict(
        # use quadratic formula to warm up 5 epochs
        # and lr is updated by iteration
        # TODO: fix default scope in get function
        type='mmdet.QuadraticWarmupLR',
        by_epoch=True,
        begin=0,
        end=warmup_epochs,
        convert_to_iter_based=True),
    dict(
        # use cosine lr from 5 to 285 epoch
        type='CosineAnnealingLR',
        eta_min=base_lr * lr_factor,
        begin=warmup_epochs,
        T_max=max_epochs - num_last_epochs,
        end=max_epochs - num_last_epochs,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        # use fixed lr during last 15 epochs
        type='ConstantLR',
        by_epoch=True,
        factor=1,
        begin=max_epochs - num_last_epochs,
        end=max_epochs,
    )
]
