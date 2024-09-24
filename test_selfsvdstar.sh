# generate desmoking results
bash ./tools/test.sh ./configs/dehazers/selfsvd/selfsvdstar.py ./work_dirs/selfsvdstar/selfsvdstar.pth ./work_dirs/selfsvdstar/SelfSVDstar 1
# calculate metrics
CUDA_VISIBLE_DEVICES=0 python ./cal_metrics.py --result_dir ./work_dirs/selfsvdstar/SelfSVDstar --target_dir ../dataset/LSVD_test_pro