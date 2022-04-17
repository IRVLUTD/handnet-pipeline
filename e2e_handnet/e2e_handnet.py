from typing_extensions import OrderedDict
from cv2 import decomposeHomographyMat
import torch

from torch import inference_mode, nn, Tensor
from fcos_utils.fcos import FCOS
from mano_train.networks.handnet import HandNet

from fpn_utils.faster_rcnn_fpn import TwoMLPHead
from .bridge import E2EBridge

import torch.nn.functional as torch_f
from datasets3d.queries import BaseQueries, TransQueries
from a2j.a2j import A2JModel
import torch.nn.functional as F


#from .bridge import E2EBridge

def load_pretrained_fcos(args, reload_detector=False):
    print("Loading pretrained detector")
    detector = FCOS(num_classes=23, ext=False, nms_thresh=0.5)
    
    if reload_detector:
        checkpoint = torch.load(args.pretrained_fcos, map_location="cpu")
        detector.load_state_dict(checkpoint["model"], strict=False)
    
    for p in detector.parameters():
        p.requires_grad = False
    return detector

def load_pretrained_a2j(args, reload_a2j=False):
    print("Loading pretrained a2j")
    a2j = A2JModel(21, crop_height=176, crop_width=176)
    if reload_a2j:
        checkpoint = torch.load(args.pretrained_a2j, map_location="cpu")
        a2j.load_state_dict(checkpoint["model"], strict=False)
    for p in a2j.parameters():
        p.requires_grad = False
    return a2j

class E2EHandNet(nn.Module):
    """
    Implements End-to-End HandNet
    """

    def __init__(
        self,
        args,
        reload_detector: bool = False,
        reload_a2j: bool = False,
    ):
        super().__init__()
        self.detector = load_pretrained_fcos(args, reload_detector)
        self.detector.eval()
        self.a2j = load_pretrained_a2j(args, reload_a2j)
    
    def forward(
        self, 
        images,
        depth_images=None,
        targets=None,
        jt_uvd_gt=None,
        is_3D: bool = False,
        is_detect: bool=False,
    ):

        if is_3D:
            if self.training:
                # A2J training
                print('fill')
            else:
                # A2J inference
                print('fill')
        elif is_detect:
            if self.training:
                print('fill')
                # detector training
            else:
                 print('fill')
                # detector inference
        else:
            # ensemble inference
            final_results = torch.zeros((len(images), 21, 3))
            
            image_mask = torch.zeros((len(images)), dtype=torch.bool)

            detections = self.detector(images, None)
            images = torch.stack(images)

            hand_mask = [(res_per_image['labels'] == 22) for res_per_image in detections ]

            depth_batch = []
            crops = []
            for img_idx, det_per_image in enumerate(detections):
                boxes = det_per_image['boxes'][hand_mask[img_idx]]
                
                if len(boxes) == 0:
                    crops.append(None)
                    continue
                if len(boxes) > 1:
                    boxes = boxes[:1]
                image_mask[img_idx] = True

                box = boxes.reshape(4).to(torch.int64)

                # pad box
                w = box[2] - box[0]
                h = box[3] - box[1]
                percent = 0.3
                box[0] = max(0, box[0] - percent * (w))
                box[1] = max(0, box[1] - percent * (h))
                box[2] = min(images[img_idx].shape[2], box[2] + percent * (w))
                box[3] = min(images[img_idx].shape[1], box[3] + percent * (h))

                crops.append(box)
                try:
                    depth_crop = F.interpolate(depth_images[img_idx, :, box[1]:box[3] + 1, box[0]:box[2] + 1].unsqueeze(0), size=(176, 176)).squeeze(0)
                except:
                    print("hi")
                depth_batch.append(depth_crop)

            if len(depth_batch) == 0:
                return final_results, torch.zeros_like(depth_images), torch.zeros((len(images), 4))

            depth_batch = torch.stack(depth_batch)
            crops = torch.stack(crops)

            a2j_pred = self.a2j(depth_batch)
            final_results[image_mask] = a2j_pred

            return final_results, depth_batch, crops