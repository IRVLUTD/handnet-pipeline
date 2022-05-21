
"""The data layer used during training to train an FPN Fast R-CNN network.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.utils.data as data
from PIL import Image
import torch

from PIL import Image
import numpy as np


class roiFPNbatchLoader(data.Dataset):
    def __init__(self, roidb, transform):
        self.roidb = roidb
        self.transform = transform
    def __getitem__(self, idx):
        sample = self.roidb[idx]
        im = Image.open(sample['image']).convert('RGB')

        gt_inds = np.where(sample['gt_classes'] != 0)[0]
        gt_boxes = np.empty((len(gt_inds), 4), dtype=np.float32)
        gt_boxes[:, 0:4] = sample['boxes'][gt_inds, :]
        labels = sample['gt_classes'][gt_inds]

        handinfo = np.empty((len(gt_inds), 5), dtype=np.float32)
        handinfo[:,0] = sample['contactstate'][gt_inds]
        handinfo[:,1] = sample['handside'][gt_inds]
        handinfo[:,2] = sample['magnitude'][gt_inds]
        handinfo[:,3] = sample['unitdx'][gt_inds]
        handinfo[:,4] = sample['unitdy'][gt_inds]

        img_id = sample['img_id']
        

        boxes = torch.as_tensor(gt_boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        image_id = torch.tensor([img_id])
        box_info = torch.as_tensor(handinfo, dtype=torch.float32)


        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        target = {}

        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["box_info"] = box_info

        return self.transform(im), target
    
    def __len__(self):
        return len(self.roidb)