_base_ = 'yolov8_runtime_schedule_300e.py'

max_epochs = 10

val_interval = 1
default_hooks = dict(
    param_scheduler=dict(max_epochs=max_epochs)
)

train_cfg = dict(max_epochs=max_epochs, val_interval=val_interval)

