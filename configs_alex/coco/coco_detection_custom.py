_base_ = '../_base_/datasets/coco_detection.py'

data_root = '/home/alex/data/coco/'
train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    dataset=dict(data_root=data_root)
)

val_dataloader = dict(
    dataset=dict(data_root=data_root)
)

test_dataloader = val_dataloader


val_evaluator = dict(
    ann_file=data_root + 'annotations/instances_val2017.json'
)
test_evaluator = val_evaluator
