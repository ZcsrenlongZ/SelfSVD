
_base_ = [
    '../_base_/datasets/selfsvd_dataset.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_selfsvd.py'
]

exp_name = 'selfsvd'


# model settings
model = dict(
    type='SelfSVD',
    generator=dict(
        type='SelfSVDNet',
        mid_channels=64,
        num_blocks=60),
    discriminator=dict(
        type='PatchDiscriminator',
        in_channels=3,
        base_channels=64,
        num_conv=3,
        norm_cfg=dict(type='IN'),
        init_cfg=dict(type='normal', gain=0.02)),
    pixel_loss=dict(type='L1Loss', loss_weight=1.0, reduction='mean'),
    gan_loss=dict(
        type='GANLoss',
        gan_type='lsgan',
        real_label_val=1.0,
        fake_label_val=0.0,
        loss_weight=1.0),
)

data = dict(
    train_dataloader=dict(samples_per_gpu=4, drop_last=True)
)

# runtime settings
work_dir = f'./work_dirs/{exp_name}'

# optimizer
optimizers = dict(
    generator=dict(
        type='Adam',
        lr=1e-4,  
        betas=(0.9, 0.99),
        paramwise_cfg=dict(custom_keys={'alignnet':dict(lr_mult=0.25)})),
    discriminator=dict(type='Adam', lr=2e-4, betas=(0.9, 0.99)))    

# learning policy
lr_config = dict(
    policy='CosineRestart',
    by_epoch=False,
    periods=[100000],
    restart_weights=[1],
    min_lr=1e-7)

# model training and testing settings
train_cfg = None
test_cfg = None
visual_config = None
