# Custom dataset: ~8k labeled + ~40k unlabeled, 2 GPUs x 2 imgs, 160k iters.
# Inherits pipelines (base_dino_detr_ssod_voc.py) and model (dino_detr_ssod_r50_voc_80k.py);
# only the dataset, class count and schedule are overridden here.
_base_ = "base_dino_detr_ssod_voc.py"

data_root = "/path/to/mydata/"
classes = ('class_a', 'class_b', 'class_c')  # must match "categories" order in the json files
num_classes = 3                               # = len(classes)

max_iters = 160000
warm_up_step = 80000  # one-to-many -> one-to-one switch at max_iters / 2

model = dict(
    # backbone=dict(with_cp=True),  # uncomment if you hit OOM (saves memory, ~20% slower)
    bbox_head=dict(num_classes=num_classes),
    train_cfg=dict(warm_up_step=warm_up_step),
    test_cfg=dict(warm_up_step=warm_up_step),
)

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        sup=dict(
            type="CocoDataset",
            ann_file=data_root + "annotations/labeled.json",
            img_prefix=data_root + "images/labeled/",
            classes=classes,
        ),
        unsup=dict(
            type="CocoDataset",
            ann_file=data_root + "annotations/unlabeled.json",
            img_prefix=data_root + "images/unlabeled/",
            classes=classes,
        ),
    ),
    val=dict(
        type="CocoDataset",
        ann_file=data_root + "annotations/val.json",
        img_prefix=data_root + "images/val/",
        classes=classes,
    ),
    test=dict(
        type="CocoDataset",
        ann_file=data_root + "annotations/val.json",
        img_prefix=data_root + "images/val/",
        classes=classes,
    ),
    sampler=dict(
        train=dict(
            sample_ratio=[1, 4],  # no effect at samples_per_gpu=2 (always 1 sup + 1 unsup)
        )
    ),
)

semi_wrapper = dict(
    type="DinoDetrSSOD",
    model="${model}",
    train_cfg=dict(
        use_teacher_proposal=False,
        pseudo_label_initial_score_thr=0.4,
        min_pseduo_box_size=0,
        unsup_weight=4.0,
    ),
    test_cfg=dict(inference_on="student"),
)

custom_hooks = [
    dict(type="NumClassCheckHook"),
    dict(type="MeanTeacher", momentum=0.999, interval=1, warm_up=0),
    dict(type='StepRecord', normalize=False),
]

optimizer = dict(lr=0.0001)  # drop to 0.00005 if loss spikes / NaN
lr_config = dict(step=[136000])  # single x0.1 drop at 85%
runner = dict(_delete_=True, type="IterBasedRunner", max_iters=max_iters)
evaluation = dict(type="SubModulesDistEvalHook", interval=4000)
checkpoint_config = dict(by_epoch=False, interval=4000, max_keep_ckpts=5, create_symlink=False)

work_dir = "work_dirs/${cfg_name}"
log_config = dict(
    interval=50,
    hooks=[
        dict(type="TextLoggerHook", by_epoch=False),
        dict(type='TensorboardLoggerHook', by_epoch=False),
    ],
)
