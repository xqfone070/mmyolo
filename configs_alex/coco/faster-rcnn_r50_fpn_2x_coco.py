_base_ = [
    '../_base_/models/faster-rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_2x.py', '../_base_/default_runtime.py'
]


data_root = '/home/alex/data/coco'
train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    dataset=dict(data_root=data_root)
)

val_dataloader = dict(
    dataset=dict(data_root=data_root)
)

test_dataloader = val_dataloader
