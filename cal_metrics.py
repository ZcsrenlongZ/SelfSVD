import sys 
sys.path.append("..")
import os
import glob
import mmedit.utils.nriqa as nriqa
from mmedit.models.dehazers.basic_dehazer import ALignPSNR, ALignSSIM
from mmedit.core import psnr, ssim, tensor2img
import numpy as np
import torch
import cv2
import argparse


def cal_metrics(result_dir, target_dir):
    results_dir = result_dir
    test_dir =  target_dir
    ids_results = os.listdir(results_dir)
    ids_test = os.listdir(test_dir)

    assert ids_results == ids_test
    ids = ids_results

    align_psnr = ALignPSNR().cuda()
    align_ssim = ALignSSIM().cuda()

    psnr_list = []
    ssim_list = []
    for id in ids:
        psnr_id = []
        ssim_id = []
        paths_out = sorted(glob.glob(os.path.join(results_dir, id, '*.png')), key=lambda f: int(f.split('/')[-1].split('.')[0]))
        paths_gt = sorted(glob.glob(os.path.join(test_dir, id, '*.png')), key=lambda f: int(f.split('/')[-1].split('.')[0]))[0:1] * len(paths_out)

        for idx in range(0, len(paths_out)):
            path_out = paths_out[idx]
            path_gt = paths_gt[idx]

            gt = cv2.cvtColor(cv2.imread(path_gt, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)/255.
            gt = np.transpose(gt, (2, 0,1))
            gt =  torch.FloatTensor(gt).unsqueeze(dim=0).cuda()

            out = cv2.cvtColor(cv2.imread(path_out, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)/255.
            out = np.transpose(out, (2, 0,1))
            out =  torch.FloatTensor(out).unsqueeze(dim=0).cuda()

            psnr_id.append(align_psnr(out, gt, 0, None))
            ssim_id.append(align_ssim(out, gt, 0, None))

        print(id, np.mean(psnr_id), np.mean(ssim_id))

        psnr_list.append(np.mean(psnr_id))
        ssim_list.append(np.mean(ssim_id))

    print(f'{results_dir}, {test_dir}, Aligned PSNR:{np.mean(psnr_list)},  Aligned SSIM:{np.mean(ssim_list)}')


    nriqa.main(results_dir, 0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="evaluate desmoking results")
    parser.add_argument("--result_dir", type=str, default="", help="desmoking result path")
    parser.add_argument("--target_dir", type=str, default='../dataset/LSVD_test_pro', help="desmoking result path")

    args = parser.parse_args()

    cal_metrics(args.result_dir, args.target_dir)










