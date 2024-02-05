_base_ = 'yolov8_runtime_schedule_300e.py'

max_epochs = 100
interval = 5
default_hooks = dict(
    checkpoint=dict(interval=interval),
    param_scheduler=dict(max_epochs=max_epochs)
)


train_cfg = dict(max_epochs=max_epochs, val_interval=interval)
