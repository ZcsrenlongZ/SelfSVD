train_dataset_type = 'SelfSVDDataset'
test_dataset_type = 'SelfSVDDataset'

img_norm_cfg_lq = dict(
    mean=[0., 0., 0.], std=[1., 1., 1.], to_rgb=True
)
img_norm_cfg_gt = dict(
    mean=[0., 0., 0.], std=[1., 1., 1.], to_rgb=True,
)
crop_size = 256
num_input_frames = 5

io_backend = 'disk'
load_kwargs = dict()

train_pipeline = [
    dict(type='GenerateSegmentIndicesDewaterMaskFour', interval_list=[1], filename_tmpl='{:08d}.png'),
    dict(type='LoadImageFromFileList',
         io_backend=io_backend,
         key='lq',
         flag='unchanged',
         **load_kwargs),
    dict(type='LoadImageFromFileList',
         io_backend=io_backend,
         key='gt',
         flag='unchanged',
         **load_kwargs),
    dict(type='LoadImageFromFileList',
         io_backend=io_backend,
         key='mask',
         flag='unchanged',
         **load_kwargs),
    dict(type='LoadImageFromFileList',
         io_backend=io_backend,
         key='refmask',
         flag='unchanged',
         **load_kwargs),
    dict(type='RescaleToZeroOne', keys=['lq', 'gt', 'mask', 'refmask']),
    dict(type='Normalize',
         keys=['lq'],
         **img_norm_cfg_lq),
    dict(type='Normalize',
         keys=['gt'],
         **img_norm_cfg_gt),
    dict(type='Normalize',
         keys=['mask'],
         **img_norm_cfg_gt),
    dict(type='QuadRandomCrop', gt_patch_size=crop_size),
    dict(type='Flip', keys=['lq', 'gt', 'mask', 'refmask'], flip_ratio=0.5,
         direction='horizontal'),
    dict(type='FramesToTensor', keys=['lq', 'gt', 'mask', 'refmask']),
    dict(type='Collect', keys=['lq', 'gt', 'mask','refmask'], meta_keys=['lq_path', 'gt_path', 'key']),
]
test_pipeline = [
    dict(type='GenerateSegmentIndicesDewaterMaskFour', interval_list=[1], filename_tmpl='{:08d}.png'),
    dict(type='LoadImageFromFileList',
         io_backend=io_backend,
         key='lq',
         flag='unchanged',
         **load_kwargs),
    dict(type='LoadImageFromFileList',
         io_backend=io_backend,
         key='gt',
         flag='unchanged',
         **load_kwargs),
    dict(type='LoadImageFromFileList',
         io_backend=io_backend,
         key='mask',
         flag='unchanged',
         **load_kwargs),
    dict(type='LoadImageFromFileList',
         io_backend=io_backend,
         key='refmask',
         flag='unchanged',
         **load_kwargs),
    dict(type='RescaleToZeroOne', keys=['lq', 'gt', 'mask', 'refmask']),
    dict(type='Normalize',
         keys=['lq'],
         **img_norm_cfg_lq),
    dict(type='Normalize',
         keys=['gt'],
         **img_norm_cfg_gt),
    dict(type='Normalize',
         keys=['mask'],
         **img_norm_cfg_gt),
    dict(type='FramesToTensor', keys=['lq', 'gt', 'mask', 'refmask']),
    dict(type='Collect', keys=['lq', 'gt', 'mask','refmask'], meta_keys=['lq_path', 'lq_path', 'key']),
]

data = dict(
    workers_per_gpu=6,
    train_dataloader=dict(samples_per_gpu=4, drop_last=True),
    val_dataloader=dict(samples_per_gpu=1, workers_per_gpu=2),
    test_dataloader=dict(samples_per_gpu=1, workers_per_gpu=2),

    train=dict(
        type='RepeatDataset',
        times=10000,
        dataset=dict(
            type=train_dataset_type,
            lq_folder='../dataset/LSVD_train',
            gt_folder='../dataset/LSVD_train',
            mask_folder='../dataset/LSVD_train',     # ref path
            refmask_folder='../dataset/LSVD_train_mask',   # ref mask path
            ann_file = None,
            num_input_frames=num_input_frames,
            pipeline=train_pipeline,
            img_extension='.png',
            scale=1,
            test_mode=False)),
    val=dict(
        type=test_dataset_type,
        lq_folder='../dataset/LSVD_test',
        gt_folder='../dataset/LSVD_test',
        mask_folder='../dataset/LSVD_test',
        refmask_folder='../dataset/LSVD_test_mask',
        ann_file = None,
        pipeline=test_pipeline,
        img_extension='.png',
        scale=1,
        test_mode=True),
    test=dict(
        type=test_dataset_type,
        lq_folder='../dataset/LSVD_test',
        gt_folder='../dataset/LSVD_test',
        mask_folder='../dataset/LSVD_test',
        refmask_folder='../dataset/LSVD_test_mask',
        ann_file = None,
        pipeline=test_pipeline,
        img_extension='.png',
        scale=1,
        test_mode=True)
)
