import math
_base_ = 'faster-rcnn_yolov8_bone_neck_head.py'


def make_div(x, factor, div=8):
    return math.ceil(x * factor / div) * div


deepen_factor = 1.00
widen_factor = 1.00
last_stage_out_channels = 512

neck_out_channels = 256
neck_real_out_channels = make_div(neck_out_channels, widen_factor)

model = dict(
    backbone=dict(
        last_stage_out_channels=last_stage_out_channels,
        deepen_factor=deepen_factor,
        widen_factor=widen_factor),
    neck=dict(
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        in_channels=[256, 512, last_stage_out_channels],
        out_channels=[256, 512, last_stage_out_channels],
        final_out_channels=neck_out_channels),
    bbox_head=dict(
        head_module=dict(
            widen_factor=widen_factor,
            in_channels=[neck_out_channels] * 3)),
    roi_head=dict(
        bbox_roi_extractor=dict(
            out_channels=neck_real_out_channels,
        ),
        bbox_head=dict(
            in_channels=neck_real_out_channels,
        )
    )
)
