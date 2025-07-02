#!/usr/bin/python
# -*- encoding: utf-8 -*-

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

import os.path as osp
import os
from PIL import Image
import numpy as np
import json
import cv2

from transform import *

IMG_DIR = "isolated_images_with_holes_topology"
MASK_DIR = "masks_single_channel"

# `FaceMask` class name is an artifact of the repository on which code was based 
# https://github.com/zllrunning/face-parsing.PyTorch

class FaceMask(Dataset): 
    def __init__(self, rootpth="/workspace/bisenet_training/nilay_png_annotation_data/labellerr_dataset", cropsize=(640, 480), mode='train', *args, **kwargs):
        super(FaceMask, self).__init__(*args, **kwargs)
        assert mode in ('train', 'val', 'test')
        self.mode = mode
        self.ignore_lb = 255
        self.rootpth = rootpth

        self.imgs = os.listdir(os.path.join(self.rootpth, IMG_DIR))

        #  pre-processing
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
        self.trans_train = Compose([
            ColorJitter(
                brightness=0.5,
                contrast=0.5,
                saturation=0.5),
            HorizontalFlip(),
            RandomScale((0.75, 1.0, 1.25, 1.5, 1.75, 2.0)),
            RandomCrop(cropsize)
            ])

    def __getitem__(self, idx):
        impth = self.imgs[idx]
        # print(f"impth: {impth}, impth[:-3]: {os.path.splitext(impth)[0]}")
        img = Image.open(osp.join(self.rootpth, IMG_DIR, impth))
        img = img.resize((512, 512), Image.BILINEAR)
        label = Image.open(osp.join(self.rootpth, MASK_DIR, os.path.splitext(impth)[0]+'_single_channel.png')).convert('P')
        label = label.resize((512, 512), Image.NEAREST)
        # print(np.unique(np.array(label)))
        if self.mode == 'train':
            im_lb = dict(im=img, lb=label)
            im_lb = self.trans_train(im_lb)
            img, label = im_lb['im'], im_lb['lb']
        img = self.to_tensor(img)
        label = np.array(label).astype(np.int64)[np.newaxis, :]
        return img, label, impth

    def __len__(self):
        return len(self.imgs)


