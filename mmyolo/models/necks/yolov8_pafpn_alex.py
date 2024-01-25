from typing import List, Union

import torch
import torch.nn as nn
from mmdet.utils import ConfigType, OptMultiConfig
from mmcv.cnn import ConvModule
from mmyolo.registry import MODELS
from ..utils import make_divisible
from .yolov8_pafpn import YOLOv8PAFPN


@MODELS.register_module()
class YOLOv8PAFPNAlex(YOLOv8PAFPN):
    def __init__(self,
                 in_channels: List[int],
                 out_channels: Union[List[int], int],
                 final_out_channels: int,
                 deepen_factor: float = 1.0,
                 widen_factor: float = 1.0,
                 num_csp_blocks: int = 3,
                 freeze_all: bool = False,
                 norm_cfg: ConfigType = dict(
                     type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='SiLU', inplace=True),
                 init_cfg: OptMultiConfig = None):

        self.final_out_channels = final_out_channels
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            deepen_factor=deepen_factor,
            widen_factor=widen_factor,
            num_csp_blocks=num_csp_blocks,
            freeze_all=freeze_all,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            init_cfg=init_cfg)

    # rewrite by alex
    def build_out_layer(self, idx: int):
        return ConvModule(
            make_divisible(self.out_channels[idx], self.widen_factor),
            make_divisible(self.final_out_channels, self.widen_factor),
            1,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

