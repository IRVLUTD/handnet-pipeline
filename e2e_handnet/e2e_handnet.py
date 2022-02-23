from typing_extensions import OrderedDict
import torch
from typing import Callable, Dict, List, Tuple, Optional

from torch import nn, Tensor
from fcos_utils.fcos import FCOS
from mano_train.networks.handnet import HandNet

from fpn_utils.faster_rcnn_fpn import TwoMLPHead
from .bridge import E2EBridge

import torch.nn.functional as torch_f
from datasets3d.queries import BaseQueries, TransQueries


#from .bridge import E2EBridge

def load_pretrained_fcos(args):
    print("Loading pretrained detector")
    detector = FCOS(num_classes=23, ext=False, e2e=True)
    checkpoint = torch.load(args.pretrained, map_location="cpu")
    detector.load_state_dict(checkpoint["model"], strict=False)
    
    for p in detector.parameters():
        p.requires_grad = False
    return detector

# def load_pretrained_fpn(args):
#     print("Loading pretrained detector")
#     detector = 
#     detector.load_state_dict(torch.load(args.pretrained), strict=False)
#     return detector


class E2EHandNet(nn.Module):
    """
    Implements End-to-End HandNet
    """

    def __init__(
        self,
        args,
    ):
        super().__init__()
        self.detector = load_pretrained_fcos(args)
        # self.detector.eval()

        #self.detector = load_pretrained_fpn(args)
        #self.detector.eval()

        self.handnet_single = HandNet(
                1024,
                mano_neurons=args.hidden_neurons,
                mano_root="misc/mano",
                mano_comps=args.mano_comps,
                mano_use_pca=args.mano_use_pca,
                mano_use_shape=args.mano_use_shape,
                mano_use_joints2d=args.mano_use_joints2d,
        )

        self.handnet = torch.nn.DataParallel(self.handnet_single)
        self.bridge = E2EBridge(out_channels=self.detector.backbone.out_channels)
    
    def forward(
        self, 
        images: List[Tensor],
        sample: Optional[List[Dict[str, Tensor]]] = None,
        single: bool = False,
    ):
        detections, image_shapes, original_image_shapes, features = self.detector(images, None) # N x Dict[str, Tensor]
        
        hand_mask = [(det['labels'] == 22) for det in detections ]

        feature_idx = []
        sides = []
        for det_idx, det in enumerate(detections):
            feature_idx.append(det['feature_idx'][hand_mask[det_idx]])
            sides.append(det['sides'][hand_mask[det_idx]])

        boxes = [
            [ det['boxes'][hand_mask[i]][ torch.where( feature_idx[i] == feat_idx) ] for i, det in enumerate(detections) ]
            for feat_idx in range(len(features))
        ]

        target, batch_idx, level_idx = self.bridge(sample, features, boxes, image_shapes, sides)

        results = self.handnet(target)

        if self.training:

            # # compute losses
            model_losses = {}
            model_loss = None

            results_names_dict = {
                "verts": TransQueries.verts3d,
                "joints": TransQueries.joints3d,
                "joints2d": TransQueries.joints2d,
            }

            for r_name in results_names_dict.keys():
                gt = target[results_names_dict[r_name]].cuda()
                model_losses[r_name] = torch_f.mse_loss(
                    results[r_name], gt.float()
                ).float()

            # shape regularizer to avg shape: 0 [Nx10]
            model_losses["shape_reg"] = torch_f.mse_loss(
                results["shape"], torch.zeros_like(results["shape"])
            ).float()

            model_loss = 0.167*sum(model_losses.values())
            model_losses['total_loss'] = model_loss

            return model_losses, results, target
        else:
            return results, detections, batch_idx, level_idx, boxes

            # original_image_sizes: List[Tuple[int, int]] = []
            # for img in images:
            #     val = img.shape[-2:]
            #     assert len(val) == 2
            #     original_image_sizes.append((val[0], val[1]))


            # images, _ = self.detector.transform(images, None)
            # with torch.no_grad():
            #     features = self.detector.backbone(images.tensors)
            # features.pop('pool', None)
            # image_shapes = images.image_sizes

            # sample["box"] = self.detector.postprocess_single(sample["box"], original_image_sizes[0], image_shapes[0])

            # boxes = [
            #     sample["box"].cuda()
            #     for feat_idx in range(len(features))
            # ]
            # target, batch_idx, level_idx = self.bridge(sample, features, boxes, image_shapes, None)

            # results = self.handnet(target)