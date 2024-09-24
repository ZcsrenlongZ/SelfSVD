import torch
import torch.nn.functional as F
import einops
import numpy as np
import cv2
import os

from ..registry import MODELS
from .basic_dehazer import BasicDehazer

from mmedit.models.losses.pixelwise_loss import l1_loss
from mmedit.models.common.pwcnet import PWCNet
from ..builder import build_backbone, build_component, build_loss
from ..common import set_requires_grad
from mmedit.utils import get_root_logger
from collections import OrderedDict
import torch.nn as nn

torch.autograd.set_detect_anomaly(True)

@MODELS.register_module()
class SelfSVD(BasicDehazer):
    def __init__(self,
                 generator,
                 pixel_loss,
                 discriminator=None,
                 gan_loss=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super().__init__(generator, pixel_loss, train_cfg, test_cfg,
                         pretrained)
        self.alignnet = PWCNet(load_pretrained=True, weights_path="./pretrained_dirs/pwcnet-network-default.pth")
        self.gan_loss = GANLoss(real_label_val=1.0, fake_label_val=0.0, loss_weight=1.0)
        self.discriminator = build_component(discriminator)
        self.discriminator.init_weights(pretrained=None)

    @staticmethod
    def _get_output_from_dict(x):
        if isinstance(x, dict):
            return x['out']
        return x

    def forward_train(self, lq, gt):
        return dict(lq=lq['lqs'], ref=lq['refs'], refmask=lq['refmask'], gt=gt)

    def forward_test(self,
                     lq,
                     gt=None,
                     meta=None,
                     save_image=False,
                     save_path=None,
                     iteration=None):
        torch.cuda.empty_cache()
        refmask = lq['refmask']
        lq = lq['lqs']

        with torch.no_grad():
            output = self.forward_segment_test(lq, gt, refmask)

        if self.test_cfg is not None and self.test_cfg.get('metrics', None):
            assert gt is not None, (
                'evaluation with metrics must have gt images.')
            results = dict(eval_result=self.evaluate(output, gt))
        else:
            results = dict(lq=lq.cpu(), output=output.cpu())
            if gt is not None:
                results['gt'] = gt.cpu()
        # save image
        if save_image:
            self._save_image(output, meta, save_path, iteration)

        return results

    def forward_segment_test(self, lq, gt, refmask):
        s_length = 5
        b, t, c, h, w = lq.shape
        with torch.no_grad():
            if t <= s_length:
                return self._get_output_from_dict(self.generator({'lq':lq, 'lq_ref':gt, 'refmask':refmask, 'is_train':False}))
            else:
                results = torch.zeros_like(lq)
                for i in range(0, t, s_length):
                    results[:, i:i+s_length, ...] = self._get_output_from_dict(self.generator({'lq':lq[:, i:i+s_length, ...],
                                                                                                'lq_ref':gt[:, i:i+s_length, ...],
                                                                                                'refmask':refmask[:, i:i+s_length, ...],
                                                                                                'is_train':False}))
                results[:, t-s_length:t, ...] = self._get_output_from_dict(self.generator({'lq':lq[:, t-s_length:t, ...],
                                                                                            'lq_ref':gt[:, t-s_length:t, ...],
                                                                                            'refmask':refmask[:, t-s_length:t, ...],
                                                                                            'is_train':False}))
                return results

    def train_step(self, data_batch, optimizer):
        data = self(**data_batch, test_mode=False)
        lq = data['lq']
        ref= data['ref']
        gt = data['gt']
        refmask = data['refmask']

        b, t, c, h, w = gt.shape
        assert lq.ndim == 5 and lq.shape[1] > 1, f"Video dehazing methods should have input t > 1 but get: {lq.shape}"
        losses = dict()
        output = self.generator({"lq":lq, "lq_ref":ref, "refmask":refmask, 'is_train':True})

        
        loss_name = None
        if isinstance(output, dict):
            # regularization loss
            ref_feat = output['ref_feat']
            losses[f'loss_l1_ref'] = self.pixel_loss(ref_feat, torch.zeros_like(ref_feat))

            output = self._get_output_from_dict(output)
            aligned_pred, mask, _ = self.alignnet(output.view(-1, c, h,w), gt.view(-1, c, h, w), scale=1.0)
            aligned_pred = aligned_pred.view(b, t, c, h, w)
            mask = mask.view(b, t, -1, h, w)

            # reconstruction loss
            losses[f'loss_l1_gt'] = self.pixel_loss(aligned_pred*mask, gt*mask)

            # ganloss loss
            losses[f'loss_d_gt'] = self.gan_loss(self.discriminator(gt.view(-1, c, h, w).detach()), True, alpha=None)+self.gan_loss(self.discriminator(output.view(-1, c, h, w).detach()), False, alpha=None)
            losses[f'loss_g_gt'] = self.gan_loss(self.discriminator(output.view(-1, c, h,w)), True, alpha=None)
 
        outputs = dict(
            losses=losses,
            num_samples=len(gt.data),
            results=dict(lq=lq.cpu(), gt=gt.cpu(), output=output.cpu()))

        losses = outputs['losses']

        # optimize generator
        optimizer['generator'].zero_grad()
        loss2 =  (losses['loss_g_gt'] + losses['loss_l1_gt']) + losses[f'loss_l1_ref']*0.05
        loss2.backward()
        optimizer['generator'].step()

        # optimize discriminator
        optimizer['discriminator'].zero_grad()
        loss1 = (losses['loss_d_gt'])
        loss1.backward()
        optimizer['discriminator'].step()
        _, log_vars = self.parse_losses(losses)
        outputs.update({'log_vars': log_vars})
        return outputs

class GANLoss(nn.Module):
    def __init__(self,
                 gan_type='lsgan',
                 real_label_val=1.0,
                 fake_label_val=0.0,
                 loss_weight=1.0):
        super().__init__()
        self.gan_type = gan_type
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val
        self.loss_weight = loss_weight
        self.loss = nn.MSELoss(reduction="none")

    def get_target_label(self, input, target_is_real):
        """Get target label.

        Args:
            input (Tensor): Input tensor.
            target_is_real (bool): Whether the target is real or fake.

        Returns:
            (bool | Tensor): Target tensor. Return bool for wgan, otherwise,
                return Tensor.
        """

        if self.gan_type == 'wgan':
            return target_is_real
        target_val = (
            self.real_label_val if target_is_real else self.fake_label_val)
        return input.new_ones(input.size()) * target_val

    def forward(self, input, target_is_real, is_disc=False, alpha=None):
        target_label = self.get_target_label(input, target_is_real)
        if alpha is None:
            loss = torch.mean(self.loss(input, target_label))
        else:
            alpha = F.interpolate(alpha, size=(30, 30), mode='bilinear', align_corners=True)
            loss = torch.mean(self.loss(input, target_label)*alpha)
        return loss * self.loss_weight