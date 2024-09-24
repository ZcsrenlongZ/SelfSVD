# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import load_checkpoint

from mmedit.models.backbones.basic_backbones.basicvsr_net import (
    ResidualBlocksWithInputConv)
from mmedit.models.common import PixelShufflePack, flow_warp
from mmedit.models.registry import BACKBONES
from mmedit.utils import get_root_logger
import torch.nn.init as init

from mmedit.models.common.pwcnet import PWCNet
from mmedit.models.common.flow_warp import flow_warp
from mmedit.models.common import set_requires_grad

@BACKBONES.register_module()
class SelfSVDNet(nn.Module):
    def __init__(self,
                 mid_channels=64,
                 num_blocks=15):
        super().__init__()
        self.mid_channels = mid_channels

        # optical flow
        self.pwcnet = PWCNet(load_pretrained=True, weights_path="./pretrained_dirs/pwcnet-network-default.pth")
        set_requires_grad(self.pwcnet, False)

        # feature extraction module
        self.feat_extract = nn.Sequential(
            nn.Conv2d(3, mid_channels, 3, 2, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(mid_channels, mid_channels, 3, 2, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            ResidualBlocksWithInputConv(mid_channels, mid_channels, 5))
        
        self.ref_feat_extract = nn.Sequential(
            nn.Conv2d(3, mid_channels, 3, 2, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(mid_channels, mid_channels, 3, 2, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            ResidualBlocksWithInputConv(mid_channels, mid_channels, 5))

        # propagation branches
        self.backbone = nn.ModuleDict()
        modules = ['forward_1']
        for i, module in enumerate(modules):
            self.backbone[module] = ResidualBlocksWithInputConv(
                (3 + i) * mid_channels, mid_channels, num_blocks)

        # upsampling module
        self.reconstruction = ResidualBlocksWithInputConv(
            2 * mid_channels, mid_channels, 5)
        self.upsample1 = PixelShufflePack(
            mid_channels, mid_channels, 2, upsample_kernel=3)
        self.upsample2 = PixelShufflePack(
            mid_channels, 64, 2, upsample_kernel=3)
        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)
        self.img_upsample = nn.Upsample(
            scale_factor=4, mode='bilinear', align_corners=False)

        self.zero_conv = nn.Conv2d(3, 3, 1, 1, 0)
        init.constant_(self.zero_conv.bias, 0)
        init.constant_(self.zero_conv.weight, 0)  

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)


    def compute_flow(self, lqs):
        n, t, c, h, w = lqs.size()
        if t == 1:
            return torch.zeros(n, 0, 2, h, w).to(lqs.device)
        lqs_1 = lqs[:, :-1, :, :, :].reshape(-1, c, h, w)
        lqs_2 = lqs[:, 1:, :, :, :].reshape(-1, c, h, w)
        flows = self.pwcnet(lqs_1,lqs_2, scale=1.0)[-1]
        flows = flows.reshape(n, t-1, 2, h, w)
        return flows

    def propagate(self, feats, flows, module_name, feats_ref_warped):
        n, t, _, h, w = flows.size()

        frame_idx = range(0, t + 1)
        flow_idx = range(-1, t)
        mapping_idx = list(range(0, len(feats['spatial'])))
        mapping_idx += mapping_idx[::-1]

        if 'backward' in module_name:
            frame_idx = frame_idx[::-1]
            flow_idx = frame_idx

        feat_prop = flows.new_zeros(n, self.mid_channels, h, w)
        for i, idx in enumerate(frame_idx):
            feat_current = feats['spatial'][mapping_idx[idx]]
            feat_ref = feats_ref_warped[:, mapping_idx[idx], ...]
            if i > 0:
                flow_n1 = flows[:, flow_idx[i], :, :, :].permute(0, 2, 3, 1)
                feat_prop = flow_warp(feat_prop, flow_n1)
            feat = [feat_current] + [feat_prop] + [feat_ref] 
            feat = torch.cat(feat, dim=1)
            feat_prop = feat_prop + self.backbone[module_name](feat)
            feats[module_name].append(feat_prop)

        if 'backward' in module_name:
            feats[module_name] = feats[module_name][::-1]
        return feats

    def upsample(self, lqs, feats):
        outputs = []
        num_outputs = len(feats['spatial'])

        mapping_idx = list(range(0, num_outputs))
        mapping_idx += mapping_idx[::-1]

        for i in range(0, lqs.size(1)):
            hr = [feats[k].pop(0) for k in feats if k != 'spatial']
            hr.insert(0, feats['spatial'][mapping_idx[i]])
            hr = torch.cat(hr, dim=1)

            hr = self.reconstruction(hr)
            hr = self.lrelu(self.upsample1(hr))
            hr = self.lrelu(self.upsample2(hr))
            hr = self.lrelu(self.conv_hr(hr))
            hr = self.zero_conv(self.conv_last(hr))
            hr += lqs[:, i, :, :, :]   

            outputs.append(hr)

        return {'out':torch.stack(outputs, dim=1)}

    def forward(self, lqs):
        ref = lqs['lq_ref']
        refmask = lqs['refmask']
        lqs = lqs['lq']
        n, t, c, h, w = lqs.size()

        lqs_downsample = F.interpolate(
            lqs.view(-1, c, h, w), scale_factor=0.25,
            mode='bicubic').view(n, t, c, h // 4, w // 4)
        
        ref_downsample = F.interpolate(
            ref.view(-1, c, h, w), scale_factor=0.25,
            mode='bicubic').view(n, t, c, h // 4, w // 4)

        feats = {}
        feats_ = self.feat_extract(lqs.view(-1, c, h, w))
        h, w = feats_.shape[2:]
        feats_ = feats_.view(n, t, -1, h, w)
        feats['spatial'] = [feats_[:, i, :, :, :] for i in range(0, t)]

        h, w = ref.shape[3:]
        ref_feat = self.ref_feat_extract(ref.view(-1,3,h,w))
        h, w = ref_feat.shape[2:]
        ref_flow = self.pwcnet(ref_downsample.view(-1, 3, h, w),
                               lqs_downsample.view(-1, 3, h, w), scale=1.0)[-1].permute(0, 2, 3, 1)
        feats_ref_warped = flow_warp(ref_feat, ref_flow).view(n, t, -1, h, w) 

        feats_ref_warped = feats_ref_warped* (1. - refmask)

        flows = self.compute_flow(lqs_downsample)
        module = 'forward_1'
        feats[module] = []
        feats = self.propagate(feats, flows, module, feats_ref_warped)

        out_dict = self.upsample(lqs, feats)
        out_dict['ref_feat'] = ref_feat
        return out_dict

    def init_weights(self, pretrained=None, strict=True):
        """Init weights for models.

        Args:
            pretrained (str, optional): Path for pretrained weights. If given
                None, pretrained weights will not be loaded. Default: None.
            strict (bool, optional): Whether strictly load the pretrained
                model. Default: True.
        """
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=strict, logger=logger)
        elif pretrained is not None:
            raise TypeError(f'"pretrained" must be a str or None. '
                            f'But received {type(pretrained)}.')

