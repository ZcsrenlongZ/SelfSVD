
total_iters = 50000
runner = dict(type='IterBasedRunner', max_iters=500000)
checkpoint_config = dict(by_epoch=False, interval=4000)
# remove gpu_collect=True in non distributed training
evaluation = dict(interval=50000, save_image=False, gpu_collect=True)
