import math
_base_ = ['mmdet::_base_/models/faster-rcnn_r50_fpn.py', ]

# The scaling factor that controls the depth of the network structure
deepen_factor = 0.33
# The scaling factor that controls the width of the network structure
widen_factor = 0.5
# The output channel of the last stage
last_stage_out_channels = 1024

# Normalization config for yolov8
norm_cfg = dict(type='BN', momentum=0.03, eps=0.001)


def make_div(x, factor, div=8):
    return math.ceil(x * factor / div) * div


bone_out_channels = [256, 512, last_stage_out_channels]
neck_in_channels = [make_div(c, widen_factor) for c in bone_out_channels]
neck_out_channels = 256
neck_real_out_channels = make_div(neck_out_channels, widen_factor)

strides = [8, 16, 32]

model = dict(
    backbone=dict(
        _delete_=True,
        _scope_='mmyolo',
        type='YOLOv8CSPDarknet',
        arch='P5',
        last_stage_out_channels=last_stage_out_channels,
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        norm_cfg=norm_cfg,
        act_cfg=dict(type='SiLU', inplace=True)),
    neck=dict(
        _delete_=True,
        _scope_='mmyolo',
        # alex修改：继承自YOLOv8PAFPN， 将neck的输出改为相同大小
        type='YOLOv8PAFPNAlex',
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        in_channels=[256, 512, last_stage_out_channels],
        out_channels=[256, 512, last_stage_out_channels],
        # alex修改：将neck的输出改为相同大小
        final_out_channels=neck_out_channels,
        num_csp_blocks=3,
        norm_cfg=norm_cfg,
        act_cfg=dict(type='SiLU', inplace=True)),
    rpn_head=dict(
        in_channels=neck_real_out_channels,
    ),
    roi_head=dict(
        bbox_roi_extractor=dict(
            featmap_strides=strides
        )
    )
)
