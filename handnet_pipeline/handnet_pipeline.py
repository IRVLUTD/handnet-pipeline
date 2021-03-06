
import torch

from torch import nn
from a2j.a2j import A2JModel
import torch.nn.functional as F

from fcos_utils.fcos import FCOS
from a2j.a2j import A2JModelLightning


#from .bridge import E2EBridge

def load_pretrained_fcos(args, reload_detector=False, num_classes=2):
    print("Loading pretrained detector")
    detector = FCOS(num_classes=num_classes, ext=False, nms_thresh=0.5) # num_class dependent on detector
    
    if reload_detector:
        checkpoint = torch.load(args.pretrained_fcos, map_location="cpu")
        detector.load_state_dict(checkpoint["model"], strict=False)
    
    for p in detector.parameters():
        p.requires_grad = False
    return detector

def load_pretrained_a2j(args, reload_a2j=False, RGBD=False):
    print("Loading pretrained a2j")
    if RGBD or 'ckpt' in args.pretrained_a2j:
        return A2JModelLightning.load_from_checkpoint(args.pretrained_a2j).eval()
    a2j = A2JModel(21, crop_height=176, crop_width=176, is_RGBD=False)
    if reload_a2j:
        checkpoint = torch.load(args.pretrained_a2j, map_location="cpu")
        a2j.load_state_dict(checkpoint["model"], strict=False)
    for p in a2j.parameters():
        p.requires_grad = False
    return a2j

class HandNet(nn.Module):
    """
    Implements End-to-End HandNet
    """

    def __init__(
        self,
        args,
        reload_detector: bool = False,
        num_classes: int = 2,
        reload_a2j: bool = False,
        RGBD: bool = False,
    ):
        super().__init__()
        self.detector = load_pretrained_fcos(args, reload_detector, num_classes)
        self.detector.eval()
        self.a2j = load_pretrained_a2j(args, reload_a2j, RGBD)
        self.RGBD = RGBD
        self.num_classes = num_classes
    
    def forward(
        self, 
        images,
        depth_images=None,
        is_3D: bool = False,
        is_detect: bool=False,
    ):
        if not is_detect and not is_3D:
            # ensemble inference
            final_results = torch.zeros((len(images), 21, 3))
            
            image_mask = torch.zeros((len(images)), dtype=torch.bool)

            detections = self.detector(images, None)
            images = torch.stack(images)

            hand_mask = [(res_per_image['labels'] == self.num_classes - 1) for res_per_image in detections ] # hand class dependent on detector

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
                percent = 0.4
                box[0] = max(0, box[0] - percent * (w))
                box[1] = max(0, box[1] - percent * (h))
                box[2] = min(images[img_idx].shape[2], box[2] + percent * (w))
                box[3] = min(images[img_idx].shape[1], box[3] + percent * (h))

                crops.append(box)
                try:
                    depth_crop = F.interpolate(depth_images[img_idx, :, box[1]:box[3] + 1, box[0]:box[2] + 1].unsqueeze(0), size=(176, 176)).squeeze(0)
                    depth_crop = depth_crop[ [2,1,0,3], :, :] if self.RGBD else depth_crop
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