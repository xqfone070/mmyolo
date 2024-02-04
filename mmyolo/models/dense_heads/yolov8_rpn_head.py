from typing import List, Optional
from torch import Tensor
from mmengine.config import ConfigDict
from mmengine.structures import InstanceData
from mmdet.utils import ConfigType
from mmyolo.registry import MODELS
from .yolov8_head import YOLOv8Head
from torch.nn.modules.batchnorm import _BatchNorm


@MODELS.register_module()
class YOLOv8RPNHead(YOLOv8Head):
    # TwoStageDetector初始化rpn的时候会传入num_classes,但是YOLOV8Head不接收该参数，所以需要进行扩展
    def __init__(self,
                 head_module: ConfigType,
                 num_classes: int = 1,
                 freeze_all: bool = False,
                 **kwargs):
        head_module.update(num_classes=num_classes)
        self.freeze_all = freeze_all
        super(YOLOv8RPNHead, self).__init__(head_module, **kwargs)

    # 作为rpn头时，传入的objectnesses固定为None
    # 默认的情况YOLOV8Head训练时，传入的objectness为bbox_dist_preds，但是predict的时候不需要改参数来计算损失
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

    def _freeze_all(self):
        """Freeze the model."""
        for m in self.modules():
            if isinstance(m, _BatchNorm):
                m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def train(self, mode=True):
        """Convert the model into training mode while keep the normalization
        layer freezed."""
        super().train(mode)
        if self.freeze_all:
            self._freeze_all()
