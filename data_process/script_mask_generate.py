import torch
from mmedit.models.common.pwcnet import PWCNet
from mmedit.models.common.pwcnet import get_backwarp
import cv2
import numpy as np
import os
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.nn as nn
from skimage.metrics import structural_similarity as ssim

def zmMinFilterGray(src, r=7):
    return cv2.erode(src, np.ones((2 * r + 1, 2 * r + 1)))

def guidedfilter(I, p, r, eps):
    height, width = I.shape
    m_I = cv2.boxFilter(I, -1, (r, r))
    m_p = cv2.boxFilter(p, -1, (r, r))
    m_Ip = cv2.boxFilter(I * p, -1, (r, r))
    cov_Ip = m_Ip - m_I * m_p

    m_II = cv2.boxFilter(I * I, -1, (r, r))
    var_I = m_II - m_I * m_I

    a = cov_Ip / (var_I + eps)
    b = m_p - a * m_I

    m_a = cv2.boxFilter(a, -1, (r, r))
    m_b = cv2.boxFilter(b, -1, (r, r))
    return m_a * I + m_b

def Defog(m, r, eps, w, maxV1):  
    V1 = np.min(m, 2) 
    Dark_Channel = zmMinFilterGray(V1, 7)

    V1 = guidedfilter(V1, Dark_Channel, r, eps)
    bins = 2000
    ht = np.histogram(V1, bins)
    d = np.cumsum(ht[0]) / float(V1.size)
    for lmax in range(bins - 1, 0, -1):
        if d[lmax] <= 0.999:
            break
    A = np.mean(m, 2)[V1 >= ht[1][lmax]].max()
    V1 = np.minimum(V1 * w, maxV1)
    return V1, A

def deHaze(m, r=81, eps=0.001, w=0.95, maxV1=0.80, bGamma=False):
    Y = np.zeros(m.shape)
    Mask_img, A = Defog(m, r, eps, w, maxV1)

    for k in range(3):
        Y[:, :, k] = (m[:, :, k] - Mask_img) / (1 - Mask_img / A)
    Y = np.clip(Y, 0, 1)
    if bGamma:
        Y = Y ** (np.log(0.5) / np.log(Y.mean()))
    return Y

def offset2distance(offset):
    delta_x = offset[:, 0, ...] 
    delta_y = offset[:, 1, ...]
    return torch.sqrt(delta_x**2+delta_y**2).cpu().numpy()


def align(clean_path, fog_path, save_path, model):
    clean = cv2.cvtColor(cv2.imread(clean_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)/255.
    clean = deHaze(clean)
    clean = np.transpose(clean, (2, 0,1))
    clean=  torch.FloatTensor(clean).unsqueeze(dim=0).cuda()
    fog = cv2.cvtColor(cv2.imread(fog_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)/255.
    fog = deHaze(fog)
    fog = np.transpose(fog, (2, 0,1))
    fog=  torch.FloatTensor(fog).unsqueeze(dim=0).cuda()
    model.eval()
    with torch.no_grad():
        fog = F.interpolate(fog, scale_factor=0.25, mode='bilinear', align_corners=True)
        clean = F.interpolate(clean, scale_factor=0.25, mode='bilinear', align_corners=True)
        offset = model(clean, fog)
        output, flow_mask = get_backwarp(clean, offset)

        output = output.cpu().detach().clamp(0., 1.).numpy().astype(np.float32)[0]
        fog = fog.cpu().detach().clamp(0., 1.).numpy().astype(np.float32)[0]
        flow_mask = flow_mask.cpu().numpy()[0]

        _, H, W = output.shape
        thre = 0.92
        PS = 8
        output1 = cv2.GaussianBlur(np.mean(output,0), (27,27), 27)
        output2 = cv2.GaussianBlur(np.mean(fog* flow_mask, 0), (27, 27), 27)
        mask = np.zeros_like(output1)
        for rr in range(0, H-PS+1, PS):
            for cc in range(0, W-PS+1, PS):
                s = ssim(output1[rr:rr+PS, cc:cc+PS], output2[rr:rr+PS, cc:cc+PS], data_range=1.)
                if s < thre:
                    mask[rr:rr+PS, cc:cc+PS] = 1
        mask = np.expand_dims(mask, 0)

        cv2.imwrite(save_path, np.transpose(mask*255,(1,2,0)).astype(np.uint8))
if __name__ == "__main__":
    model = PWCNet(load_pretrained=True, weights_path="../pretrained_dirs/pwcnet-network-default.pth").cuda()
    clean_dir = "../dataset/LSVD_train"       # "../dataset/LSVD_test" 
    save_dir = "../dataset/LSVD_train_mask"   # "../dataset/LSVD_test_mask" 
    ids = os.listdir(clean_dir)
    for id in ids:
        os.makedirs(os.path.join(save_dir, id), exist_ok=True)
        files = sorted(os.listdir(os.path.join(clean_dir, id)), key=lambda f: int(f.split('.')[0]))
        clean_path = os.path.join(clean_dir, id, files[4])
        for ii in range(0, len(files)):
            fog_path = os.path.join(clean_dir, id, files[ii])
            save_path = os.path.join(save_dir, id, fog_path.split('/')[-1])
            align(clean_path, fog_path, save_path, model)

