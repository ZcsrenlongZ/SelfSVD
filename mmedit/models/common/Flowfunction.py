"""
Functions related to optical flow
- flow_resample: Warp image using flow map
- flow_to_color: Convert optical flow to colorful image
"""
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

import numpy as np
import cv2


def get_grid(tensor):
    """
    """
    b, _, h, w = tensor.shape
    hor = torch.linspace(-1.0, 1.0, w)
    hor.requires_grad = False
    hor = hor.view(1, 1, 1, w)
    hor = hor.expand(b, 1, h, w)
    ver = torch.linspace(-1.0, 1.0, h)
    ver.requires_grad = False
    ver = ver.view(1, 1, h, 1)
    ver = ver.expand(b, 1, h, w)

    t_grid = torch.cat([hor, ver], 1)
    t_grid.requires_grad = False

    return t_grid.to(tensor.device)


def flow_resample(img, flow, normalized=False):
    """
    Warp image using flow map
    :param img: 4-D tensor
    :param flow: 4-D tensor, of size [B, 2, H, W]
    """
    b, c, h, w = img.size()

    ### For FRVSR, output normalized flow in [-1, 1]
    if normalized:
        flow = flow
        # flow = torch.cat([flow[:, 0:1, :, :], flow[:, 1:2, :, :]], dim=1)
    else:
        flow = torch.cat([flow[:, 0:1, :, :] / ((w - 1.0) / 2.0), flow[:, 1:2, :, :] / ((h - 1.0) / 2.0)], dim=1)

    grid = get_grid(img)
    final_grid = (grid + flow).permute(0, 2, 3, 1)
    img_warp = torch.nn.functional.grid_sample(img, final_grid, mode='bilinear', padding_mode='border')
    return img_warp

def make_colorwheel():
    '''
    Generates a color wheel for optical flow visualization as presented in:
        Baker et al. "A Database and Evaluation Methodology for Optical Flow" (ICCV, 2007)
        URL: http://vision.middlebury.edu/flow/flowEval-iccv07.pdf

    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun
    '''

    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3))
    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255*np.arange(0,RY)/RY)
    col = col+RY
    # YG
    colorwheel[col:col+YG, 0] = 255 - np.floor(255*np.arange(0,YG)/YG)
    colorwheel[col:col+YG, 1] = 255
    col = col+YG
    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.floor(255*np.arange(0,GC)/GC)
    col = col+GC
    # CB
    colorwheel[col:col+CB, 1] = 255 - np.floor(255*np.arange(CB)/CB)
    colorwheel[col:col+CB, 2] = 255
    col = col+CB
    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.floor(255*np.arange(0,BM)/BM)
    col = col+BM
    # MR
    colorwheel[col:col+MR, 2] = 255 - np.floor(255*np.arange(MR)/MR)
    colorwheel[col:col+MR, 0] = 255
    return colorwheel


def flow_compute_color(u, v, convert_to_bgr=False):
    '''
    Applies the flow color wheel to (possibly clipped) flow components u and v.

    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun

    :param u: np.ndarray, input horizontal flow
    :param v: np.ndarray, input vertical flow
    :param convert_to_bgr: bool, whether to change ordering and output BGR instead of RGB
    :return:
    '''

    flow_image = np.zeros((u.shape[0], u.shape[1], 3), np.uint8)

    colorwheel = make_colorwheel()  # shape [55x3]
    ncols = colorwheel.shape[0]

    rad = np.sqrt(np.square(u) + np.square(v))
    a = np.arctan2(-v, -u) / np.pi

    fk = (a + 1) / 2 * (ncols - 1)
    k0 = np.floor(fk).astype(np.int32)
    k1 = k0 + 1
    k1[k1 == ncols] = 0
    f = fk - k0

    for i in range(colorwheel.shape[1]):

        tmp = colorwheel[:,i]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1 - f) * col0 + f * col1

        idx = (rad <= 1)
        col[idx]  = 1 - rad[idx] * (1-col[idx])
        col[~idx] = col[~idx] * 0.75   # out of range?

        # Note the 2-i => BGR instead of RGB
        ch_idx = 2 - i if convert_to_bgr else i
        flow_image[:, :, ch_idx] = np.floor(255 * col)

    return flow_image


def flow_to_color(flow_uv, clip_flow=None, convert_to_bgr=False):
    '''
    Expects a two dimensional flow image of shape [H,W,2]

    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun

    :param flow_uv: np.ndarray of shape [H,W,2]
    :param clip_flow: float, maximum clipping value for flow
    :return:
    '''

    assert flow_uv.ndim == 3, 'input flow must have three dimensions'
    assert flow_uv.shape[2] == 2, 'input flow must have shape [H,W,2]'

    if clip_flow is not None:
        flow_uv = np.clip(flow_uv, 0, clip_flow)

    u = flow_uv[:,:,0]
    v = flow_uv[:,:,1]

    rad = np.sqrt(np.square(u) + np.square(v))
    rad_max = np.max(rad)

    epsilon = 1e-5
    u = u / (rad_max + epsilon)
    v = v / (rad_max + epsilon)

    return flow_compute_color(u, v, convert_to_bgr)

def flow_color(flow):
    b = flow.shape[0]
    flow_gray_b = np.zeros((b,1,flow.shape[2], flow.shape[3]))
    for i in range(b):
        flow_i = flow[i].detach().cpu().numpy()
        flow_im = flow_to_color(flow_i.transpose((1, 2, 0)))
        flow_gray = cv2.cvtColor(flow_im, cv2.COLOR_RGB2GRAY)
        flow_gray_b[i,:] = flow_gray
    return flow_gray_b


def save_flow(flow, rpath):#, graypath):
    
    flow = flow[0].cpu().detach().numpy()
    flow_im = flow_to_color(flow.transpose((1, 2, 0)))
    cv2.imwrite(rpath, cv2.cvtColor(flow_im, cv2.COLOR_RGB2BGR))
    

    
    
    
    