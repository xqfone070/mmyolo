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


# backbone about
strides = [8, 16, 32]

# neck about
neck_out_channels = 256
neck_real_out_channels = make_div(neck_out_channels, widen_factor)

# head about
rpn_loss_ratio = 0.1
loss_cls_weight = 0.5 * rpn_loss_ratio
loss_bbox_weight = 7.5 * rpn_loss_ratio
# Since the dfloss is implemented differently in the official
# and mmdet, we're going to divide loss_weight by 4.
loss_dfl_weight = 1.5 / 4 * rpn_loss_ratio

tal_topk = 10  # Number of bbox selected in each level
tal_alpha = 0.5  # A Hyper-parameter related to alignment_metrics
tal_beta = 6.0  # A Hyper-parameter related to alignment_metrics

model_test_cfg = dict(
    # The config of multi-label for multi-class prediction.
    multi_label=True,
    # The number of boxes before NMS
    nms_pre=2000,
    score_thr=0.001,  # Threshold to filter out boxes.
    nms=dict(type='nms', iou_threshold=0.7),  # NMS type and threshold
    max_per_img=1000)  # Max number of detections of each image

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
        _delete_=True,
        _scope_='mmyolo',
        type='YOLOv8RPNHead',
        head_module=dict(
            type='YOLOv8HeadModule',
            num_classes=1,
            in_channels=[neck_out_channels] * 3,
            widen_factor=widen_factor,
            reg_max=16,
            norm_cfg=norm_cfg,
            act_cfg=dict(type='SiLU', inplace=True),
            featmap_strides=strides),
        prior_generator=dict(
            type='mmdet.MlvlPointGenerator', offset=0.5, strides=strides),
        bbox_coder=dict(type='DistancePointBBoxCoder'),
        # scaled based on number of detection layers
        loss_cls=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=True,
            reduction='none',
            loss_weight=loss_cls_weight),
        loss_bbox=dict(
            type='IoULoss',
            iou_mode='ciou',
            bbox_format='xyxy',
            reduction='sum',
            loss_weight=loss_bbox_weight,
            return_iou=False),
        loss_dfl=dict(
            type='mmdet.DistributionFocalLoss',
            reduction='mean',
            loss_weight=loss_dfl_weight)),
    roi_head=dict(
        bbox_roi_extractor=dict(
            out_channels=neck_real_out_channels,
            featmap_strides=strides
        ),
        bbox_head=dict(
            in_channels=neck_real_out_channels,
        )
    ),
    # model training and testing settings
    train_cfg=dict(
        rpn=dict(
            _delete_=True,
            _scope_='mmyolo',
            assigner=dict(
                type='BatchTaskAlignedAssigner',
                num_classes=1,
                use_ciou=True,
                topk=tal_topk,
                alpha=tal_alpha,
                beta=tal_beta,
                eps=1e-9),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        rpn_proposal=model_test_cfg),
    test_cfg=dict(
        rpn=model_test_cfg
    )
)
