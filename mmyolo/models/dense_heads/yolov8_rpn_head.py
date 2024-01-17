from typing import List, Optional
from torch import Tensor
from mmengine.config import ConfigDict
from mmengine.structures import InstanceData
from mmdet.utils import ConfigType
from mmyolo.registry import MODELS
from .yolov8_head import YOLOv8Head


@MODELS.register_module()
class YOLOv8RPNHead(YOLOv8Head):
    def __init__(self,
                 head_module: ConfigType,
                 num_classes: int = 1,
                 **kwargs):
        head_module.update(num_classes=num_classes)
        super(YOLOv8RPNHead, self).__init__(head_module, **kwargs)

    def predict_by_feat(self,
                        cls_scores: List[Tensor],
                        bbox_preds: List[Tensor],
                        objectnesses: Optional[List[Tensor]] = None,
                        batch_img_metas: Optional[List[dict]] = None,
                        cfg: Optional[ConfigDict] = None,
                        rescale: bool = True,
                        with_nms: bool = True) -> List[InstanceData]:
        return super().predict_by_feat(cls_scores=cls_scores,
                                       bbox_preds=bbox_preds,
                                       objectnesses=None,
                                       batch_img_metas=batch_img_metas,
                                       cfg=cfg,
                                       rescale=rescale,
                                       with_nms=with_nms)
