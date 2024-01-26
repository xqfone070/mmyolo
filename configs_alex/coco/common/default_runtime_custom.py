import time
_base_ = 'mmdet::_base_/default_runtime.py'


default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        save_best='auto',
        max_keep_ckpts=20)
)

run_time = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))

project_name = 'default_runtime_custom'
run_name = 'default_' + run_time

wandb_init_kwargs = {'project': project_name,
                     'name': run_name}
visualizer = dict(vis_backends=[
    dict(type='LocalVisBackend'),
    dict(type='WandbVisBackend', init_kwargs=wandb_init_kwargs)
])

# Single-scale training is recommended to
# be turned on, which can speed up training.
env_cfg = dict(cudnn_benchmark=True)
