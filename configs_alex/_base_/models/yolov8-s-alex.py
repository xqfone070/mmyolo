_base_ = 'yolov8-s.py'

neck_out_channels = 256
model = dict(
    neck=dict(
        _scope_='mmyolo',
        # alex修改：继承自YOLOv8PAFPN， 将neck的输出改为相同大小
        type='YOLOv8PAFPNAlex',
        # alex修改：将neck的输出改为相同大小
        final_out_channels=neck_out_channels
    ),
    bbox_head=dict(
        head_module=dict(
            in_channels=[neck_out_channels] * 3
        )
    )
)