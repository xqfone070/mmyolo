_base_ = 'mmdet::_base_/datasets/coco_detection.py'

data_root = '/home/alex/data/coco/'


img_scale = (1333, 800)
_base_.train_pipeline[2].scale = img_scale
_base_.test_pipeline[1].scale = img_scale

train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    dataset=dict(
        data_root=data_root,
        pipeline=_base_.train_pipeline
    )
)

val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        pipeline=_base_.test_pipeline
    )
)

test_dataloader = val_dataloader


val_evaluator = dict(
    ann_file=data_root + 'annotations/instances_val2017.json'
)
test_evaluator = val_evaluator
