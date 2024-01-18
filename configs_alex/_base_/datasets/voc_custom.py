import os

dataset_type = 'YOLOv5VOCDataset'
data_root = 'data_root'  # Root path of data

img_subdir = 'images'
ann_subdir = 'annotations'
set_subdir = 'sets'
train_ann_file = os.path.join(set_subdir, 'trainval.txt')
val_ann_file = os.path.join(set_subdir, 'test.txt')

train_batch_size_per_gpu = 8
train_num_workers = 4


# dataloader
train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        img_subdir=img_subdir,
        ann_subdir=ann_subdir,
        ann_file=train_ann_file,
        data_prefix=dict(img=img_subdir, sub_data_root=''),
        filter_cfg=dict(filter_empty_gt=False, min_size=4),
        pipeline=[]
    )
)

val_dataloader = dict(
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        img_subdir=img_subdir,
        ann_subdir=ann_subdir,
        ann_file=val_ann_file,
        data_prefix=dict(img=img_subdir, sub_data_root=''),
        pipeline=[])
)

test_dataloader = val_dataloader

# evaluator
val_evaluator = dict(type='mmdet.VOCMetric', metric='mAP', eval_mode='area')

test_evaluator = val_evaluator
