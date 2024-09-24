# Copyright (c) OpenMMLab. All rights reserved.
import glob
import os
import os.path as osp

import mmcv

from .base_selfsvd_dataset import BaseSelfSVDDataset
from .registry import DATASETS


@DATASETS.register_module()
class SelfSVDDataset(BaseSelfSVDDataset):
    def __init__(self,
                 lq_folder,
                 gt_folder,
                 mask_folder,
                 refmask_folder,
                 pipeline,
                 scale,
                 ann_file=None,
                 num_input_frames=None,
                 img_extension='.png',
                 test_mode=True):
        super().__init__(pipeline, scale, test_mode)

        self.lq_folder = str(lq_folder)
        self.gt_folder = str(gt_folder)
        self.mask_folder = str(mask_folder)

        self.refmask_folder = str(refmask_folder)
        self.ann_file = ann_file

        if num_input_frames is not None and num_input_frames <= 0:
            raise ValueError('"num_input_frames" must be None or positive, '
                             f'but got {num_input_frames}.')
        self.num_input_frames = num_input_frames
        self.img_extension = img_extension

        self.data_infos = self.load_annotations()

    def _load_annotations_from_file(self):
        data_infos = []

        ann_list = mmcv.list_from_file(self.ann_file)
        for ann in ann_list:
            key, sequence_length = ann.strip().split(' ')
            if self.num_input_frames is None:
                num_input_frames = sequence_length
            else:
                num_input_frames = self.num_input_frames
            data_infos.append(
                dict(
                    lq_path=self.lq_folder,
                    gt_path=self.gt_folder,
                    key=key,
                    num_input_frames=int(num_input_frames),
                    sequence_length=int(sequence_length)))

        return data_infos

    def load_annotations(self):
        """Load annotations for the dataset.

        Returns:
            list[dict]: Returned list of dicts for paired paths of LQ and GT.
        """
        assert self.lq_folder == self.gt_folder
        assert self.ann_file is None
        if self.ann_file:
            return self._load_annotations_from_file()
            
        sequences = sorted(glob.glob(osp.join(self.lq_folder, '*'))) 
        data_infos = []
        for sequence in sequences:
            files = sorted(glob.glob(os.path.join(sequence, f'*{self.img_extension}')), key=lambda f: int(f.split('/')[-1].split('.')[0]))
            lq_files = files[1:]
            gt_files = files[0:1] * len(lq_files)
            mask_files = files[0:1] * len(lq_files) 
            refmask_files = sorted(glob.glob(os.path.join(self.refmask_folder, sequence.split('/')[-1], f'*{self.img_extension}')), key=lambda f: int(f.split('/')[-1].split('.')[0]))[5:]
            refmask_files = refmask_files[4:5] * len(lq_files)
            sequence_length = len(lq_files)
            if self.num_input_frames is None:
                num_input_frames = sequence_length
            else:
                num_input_frames = self.num_input_frames
            data_infos.append(
                dict(
                    lq_path=self.lq_folder,
                    gt_path=self.gt_folder,
                    mask_path=self.mask_folder,
                    refmask_path= self.refmask_folder,
                    lq_paths=lq_files,
                    gt_paths=gt_files,
                    mask_paths=mask_files,
                    refmask_paths=refmask_files,
                    key=sequence.replace(f'{self.lq_folder}{os.sep}', ''),
                    num_input_frames=num_input_frames,
                    sequence_length=sequence_length))

        return data_infos
