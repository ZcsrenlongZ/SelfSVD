# generate desmoking results
bash ./tools/test.sh ./configs/dehazers/selfsvd/selfsvd.py ./work_dirs/selfsvd/selfsvd.pth ./work_dirs/selfsvd/SelfSVD 1
# calculate metrics
CUDA_VISIBLE_DEVICES=0 python ./cal_metrics.py --result_dir ./work_dirs/selfsvd/SelfSVD --target_dir ../dataset/LSVD_test_pro