_base_ = [
    '../../_base_/schedules/schedule_1x.py', '../../_base_/default_runtime.py'
]

data_preprocessor = dict(
    type='DetDataPreprocessor',
    mean=[0, 0, 0],
    std=[255., 255., 255.],
    bgr_to_rgb=True,
    pad_size_divisor=32)

model = dict(
    type='YOLAWithSAM3',
    kernel_nums=8,
    kernel_size=5,
    Gtheta=[0.6, 0.8],
    loss_consistency=dict(type='SmoothL1Loss', loss_weight=0.01, reduction='sum'),
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',
        num_outs=5),
    sam3_head=dict(
        type='SAM3Adapter',
        sam3_repo_path='/home/taocheng/sam3/sam3/sam3',
        sam3_module='sam3.build',
        sam3_builder='build_sam3_detector',
        sam3_kwargs=dict(
            # 占位符：按你的 SAM3 复刻代码实际参数替换
            checkpoint='/home/taocheng/sam3/checkpoints/[YOUR_SAM3_CKPT].pth',
            device='cuda'),
        prompt_label_file='/home/taocheng/YOLA_Project/data/exdarkv3/labels.txt',
        prompt_template='detect {label} in low-light scene',
        num_classes=12),
    train_cfg=dict(),
    test_cfg=dict(
        score_thr=0.01,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=300))

dataset_type = 'ExDarkVocDataset'
data_root = '/home/taocheng/YOLA_Project/data/exdarkv3/'
backend_args = None

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Expand',
        mean=data_preprocessor['mean'],
        to_rgb=data_preprocessor['bgr_to_rgb'],
        ratio_range=(1, 2)),
    dict(
        type='MinIoURandomCrop',
        min_ious=(0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
        min_crop_size=0.3),
    dict(type='Resize', scale=(608, 608), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(608, 608), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'))
]

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='train.txt',
        data_prefix=dict(sub_data_root=''),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        backend_args=backend_args))

val_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='test.txt',
        data_prefix=dict(sub_data_root=''),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))

test_dataloader = val_dataloader
val_evaluator = dict(type='VOCMetric', metric='mAP', eval_mode='area')
test_evaluator = val_evaluator

train_cfg = dict(max_epochs=24, val_interval=1)
optim_wrapper = dict(type='OptimWrapper', optimizer=dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0005))
param_scheduler = [
    dict(type='LinearLR', start_factor=0.1, by_epoch=False, begin=0, end=1000),
    dict(type='MultiStepLR', by_epoch=True, milestones=[18, 23], gamma=0.1)
]
default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=1))
