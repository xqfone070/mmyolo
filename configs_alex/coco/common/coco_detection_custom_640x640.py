_base_ = 'coco_detection_custom_1333x800.py'

img_scale = (640, 640)
_base_.train_pipeline[2].scale = img_scale
_base_.test_pipeline[1].scale = img_scale


train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    dataset=dict(
        pipeline=_base_.train_pipeline
    )
)
