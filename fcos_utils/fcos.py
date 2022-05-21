import math
import warnings
from collections import OrderedDict
from functools import partial
from typing import Callable, Dict, List, Tuple, Optional
from matplotlib.pyplot import box

import torch
from torch import nn, Tensor

from torchvision.ops import sigmoid_focal_loss
from .utils import generalized_box_iou_loss
from torchvision.ops import boxes as box_ops
from . import det_utils
from .anchor_utils import AnchorGenerator
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.ops.feature_pyramid_network import LastLevelP6P7
import pytorch_lightning as pl


class FCOSHead(nn.Module):
    """
    A regression and classification head for use in FCOS.

    Args:
        in_channels (int): number of channels of the input feature
        num_anchors (int): number of anchors to be predicted
        num_classes (int): number of classes to be predicted
        num_convs (Optional[int]): number of conv layer of head. Default: 4.
    """

    __annotations__ = {
        "box_coder": det_utils.BoxLinearCoder,
    }

    def __init__(self, in_channels: int, num_anchors: int, num_classes: int, num_convs: Optional[int] = 4, ext: bool = True) -> None:
        super().__init__()
        self.ext = ext
        self.box_coder = det_utils.BoxLinearCoder(normalize_by_size=True)
        self.classification_head = FCOSClassificationHead(in_channels, num_anchors, num_classes, num_convs, ext=ext)
        self.regression_head = FCOSRegressionHead(in_channels, num_anchors, num_convs)

    def compute_loss(
        self,
        targets: List[Dict[str, Tensor]],
        head_outputs: Dict[str, Tensor],
        anchors: List[Tensor],
        matched_idxs: List[Tensor],
    ) -> Dict[str, Tensor]:

        cls_logits = head_outputs["cls_logits"]  # [N, HWA, C]
        bbox_regression = head_outputs["bbox_regression"]  # [N, HWA, 4]
        bbox_ctrness = head_outputs["bbox_ctrness"]  # [N, HWA, 1]
        hand_lr = head_outputs["hand_lr"]  # [N, HWA, C]

        if self.ext:
            hand_contact_state = head_outputs["hand_contact_state"]  # [N, HWA, 5C]
            hand_dxdy = head_outputs["hand_dxdy"]  # [N, HWA, 3C]


        all_gt_classes_targets = []
        all_gt_boxes_targets = []
        all_gt_hand_lr_targets = []
        if self.ext:
            all_gt_hand_contact_state_targets = []
            all_gt_hand_dxdy_targets = []
            
        for targets_per_image, matched_idxs_per_image in zip(targets, matched_idxs):
            if len(targets_per_image["labels"]) == 0:
                gt_classes_targets = targets_per_image["labels"].new_zeros((len(matched_idxs_per_image),))
                gt_boxes_targets = targets_per_image["boxes"].new_zeros((len(matched_idxs_per_image), 4))
                gt_hand_lr_targets = targets_per_image["box_info"][:, 1].new_zeros((len(matched_idxs_per_image),))
                if self.ext:
                    gt_hand_contact_state_targets = targets_per_image["hand_contact_state"].new_zeros((len(matched_idxs_per_image), 5))
                    gt_hand_dxdy_targets = targets_per_image["hand_dxdy"].new_zeros((len(matched_idxs_per_image), 3))
                    
            else:
                gt_classes_targets = targets_per_image["labels"][matched_idxs_per_image.clip(min=0)]
                gt_boxes_targets = targets_per_image["boxes"][matched_idxs_per_image.clip(min=0)]
                gt_hand_lr_targets = targets_per_image["box_info"][:, 1][matched_idxs_per_image.clip(min=0)]
                if self.ext:
                    gt_hand_contact_state_targets = targets_per_image["box_info"][:, 0][matched_idxs_per_image.clip(min=0)]
                    gt_hand_dxdy_targets = targets_per_image["box_info"][:, 2:][matched_idxs_per_image.clip(min=0)]
                    
            gt_classes_targets[matched_idxs_per_image < 0] = -1  # background
            gt_hand_lr_targets[matched_idxs_per_image < 0] = -1.  # background
            all_gt_classes_targets.append(gt_classes_targets)
            all_gt_boxes_targets.append(gt_boxes_targets)
            all_gt_hand_lr_targets.append(gt_hand_lr_targets)

            if self.ext:
                gt_hand_contact_state_targets[matched_idxs_per_image < 0] = -1.  # background
                all_gt_hand_contact_state_targets.append(gt_hand_contact_state_targets)
                all_gt_hand_dxdy_targets.append(gt_hand_dxdy_targets)
                

        all_gt_classes_targets = torch.stack(all_gt_classes_targets)
        all_gt_hand_lr_targets = torch.stack(all_gt_hand_lr_targets).long()
        if self.ext:
            all_gt_hand_contact_state_targets = torch.stack(all_gt_hand_contact_state_targets).long()
            all_gt_hand_dxdy_targets = torch.stack(all_gt_hand_dxdy_targets)
            
        # compute foregroud
        foregroud_mask = all_gt_classes_targets >= 0
        num_foreground = foregroud_mask.sum().item()

        # classification loss
        gt_classes_targets = torch.zeros_like(cls_logits)
        gt_classes_targets[foregroud_mask, all_gt_classes_targets[foregroud_mask]] = 1.0
        loss_cls = sigmoid_focal_loss(cls_logits, gt_classes_targets, reduction="sum")

        # hand lr classification loss
        hand_foreground_mask = all_gt_hand_lr_targets >= 0
        gt_hand_lr_targets = torch.zeros_like(hand_lr)
        gt_hand_lr_targets[hand_foreground_mask, all_gt_hand_lr_targets[hand_foreground_mask]] = 1.0
        loss_hand_lr = sigmoid_focal_loss(hand_lr, gt_hand_lr_targets, reduction="sum")
        loss_hand_lr *= 2e-2

        if self.ext:
            # hand contact state classification loss
            contact_state_foreground_mask = all_gt_hand_contact_state_targets >= 0
            gt_hand_contact_state_targets = torch.zeros_like(hand_contact_state)
            gt_hand_contact_state_targets[contact_state_foreground_mask, all_gt_hand_contact_state_targets[contact_state_foreground_mask]] = 1.0
            loss_hand_contact_state = sigmoid_focal_loss(hand_contact_state, gt_hand_contact_state_targets, reduction="sum")
            loss_hand_contact_state *= 1e-2

            # hand dxdy regression loss
            loss_hand_dxdy = torch.nn.functional.mse_loss(hand_dxdy, all_gt_hand_dxdy_targets)
            loss_hand_dxdy *= 10

        # regression loss: GIoU loss
        pred_boxes = [
            self.box_coder.decode_single(bbox_regression_per_image, anchors_per_image)
            for anchors_per_image, bbox_regression_per_image in zip(anchors, bbox_regression)
        ]
        # amp issue: pred_boxes need to convert float
        loss_bbox_reg = generalized_box_iou_loss(
            torch.stack(pred_boxes)[foregroud_mask].float(),
            torch.stack(all_gt_boxes_targets)[foregroud_mask],
            reduction="sum",
        )

        # ctrness loss
        bbox_reg_targets = [
            self.box_coder.encode_single(anchors_per_image, boxes_targets_per_image)
            for anchors_per_image, boxes_targets_per_image in zip(anchors, all_gt_boxes_targets)
        ]
        bbox_reg_targets = torch.stack(bbox_reg_targets, dim=0)
        if len(bbox_reg_targets) == 0:
            gt_ctrness_targets = bbox_reg_targets.new_zeros(bbox_reg_targets.size()[:-1])
        else:
            left_right = bbox_reg_targets[:, :, [0, 2]]
            top_bottom = bbox_reg_targets[:, :, [1, 3]]
            gt_ctrness_targets = torch.sqrt(
                torch.abs((left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0])
                * (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0]))
            )
        pred_centerness = bbox_ctrness.squeeze(dim=2)
        loss_bbox_ctrness = nn.functional.binary_cross_entropy_with_logits(
            pred_centerness[foregroud_mask], gt_ctrness_targets[foregroud_mask], reduction="sum"
        )

        if not math.isfinite(loss_bbox_ctrness / max(1, num_foreground)):
            print("except")

        loss_dict =  {
            "classification": loss_cls / max(1, num_foreground),
            "bbox_regression": loss_bbox_reg / max(1, num_foreground),
            "bbox_ctrness": loss_bbox_ctrness / max(1, num_foreground),
            "hand_lr": loss_hand_lr / max(1, num_foreground)
        }

        if self.ext:
            loss_dict["hand_contact_state"] = loss_hand_contact_state / max(1, num_foreground)
            loss_dict["hand_dxdy"] = loss_hand_dxdy / max(1, num_foreground)

        return loss_dict

    def forward(self, x: List[Tensor]) -> Dict[str, Tensor]:
        cls_logits, hand_contact_state, hand_dxdy, hand_lr, feature_idx = self.classification_head(x)
        bbox_regression, bbox_ctrness = self.regression_head(x)
        if self.ext:
            return {
                "cls_logits": cls_logits,
                "hand_contact_state": hand_contact_state,
                "hand_dxdy": hand_dxdy,
                "hand_lr": hand_lr,
                "feature_idx": feature_idx,
                "bbox_regression": bbox_regression,
                "bbox_ctrness": bbox_ctrness,
            }
        else:
            return {
                "cls_logits": cls_logits,
                "bbox_regression": bbox_regression,
                "bbox_ctrness": bbox_ctrness,
                "hand_lr": hand_lr,
                "feature_idx": feature_idx
            }


class FCOSClassificationHead(nn.Module):
    """
    A classification head for use in FCOS.

    Args:
        in_channels (int): number of channels of the input feature.
        num_anchors (int): number of anchors to be predicted.
        num_classes (int): number of classes to be predicted.
        num_convs (Optional[int]): number of conv layer. Default: 4.
        prior_probability (Optional[float]): probability of prior. Default: 0.01.
        norm_layer: Module specifying the normalization layer to use.
    """

    def __init__(
        self,
        in_channels: int,
        num_anchors: int,
        num_classes: int,
        num_convs: int = 4,
        prior_probability: float = 0.01,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        ext: bool = True,
    ) -> None:
        super().__init__()

        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.ext = ext

        if norm_layer is None:
            norm_layer = partial(nn.GroupNorm, 32)

        conv = []
        for _ in range(num_convs):
            conv.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1))
            conv.append(norm_layer(in_channels))
            conv.append(nn.ReLU())
        self.conv = nn.Sequential(*conv)

        for layer in self.conv.children():
            if isinstance(layer, nn.Conv2d):
                torch.nn.init.normal_(layer.weight, std=0.01)
                torch.nn.init.constant_(layer.bias, 0)

        self.cls_logits = nn.Conv2d(in_channels, num_anchors * num_classes, kernel_size=3, stride=1, padding=1)
        torch.nn.init.normal_(self.cls_logits.weight, std=0.01)
        torch.nn.init.constant_(self.cls_logits.bias, -math.log((1 - prior_probability) / prior_probability))

        self.hand_lr_layer = nn.Conv2d(in_channels, num_anchors * 2, kernel_size=3, stride=1, padding=1)
        torch.nn.init.normal_(self.hand_lr_layer.weight, std=0.01)
        torch.nn.init.zeros_(self.hand_lr_layer.bias)

        if self.ext:
            self.hand_contact_state_layer =nn.Conv2d(in_channels, num_anchors * 5, kernel_size=3, stride=1, padding=1)
            self.hand_dydx_layer = nn.Conv2d(in_channels, num_anchors * 3, kernel_size=3, stride=1, padding=1)

            
            torch.nn.init.normal_(self.hand_contact_state_layer.weight, std=0.01)
            torch.nn.init.normal_(self.hand_dydx_layer.weight, std=0.01)
        
            torch.nn.init.zeros_(self.hand_contact_state_layer.bias)
            torch.nn.init.zeros_(self.hand_dydx_layer.bias)
            

    def forward(self, x: List[Tensor]) -> Tensor:
        all_cls_logits = []
        all_feature_idx = []
        all_hand_lr = []

        if self.ext:
            all_hand_contact_state= []
            all_hand_dxdy = []
           
            
        for idx, features in enumerate(x):
            cls_tower = self.conv(features)
            cls_logits = self.cls_logits(cls_tower)

            # Permute classification output from (N, A * K, H, W) to (N, HWA, K).
            N, _, H, W = cls_logits.shape
            cls_logits = cls_logits.view(N, -1, self.num_classes, H, W)
            cls_logits = cls_logits.permute(0, 3, 4, 1, 2)
            cls_logits = cls_logits.reshape(N, -1, self.num_classes)  # Size=(N, HWA, K)

            all_cls_logits.append(cls_logits)

            handside_pred = self.hand_lr_layer(cls_tower)

            # Permute side classification output from (N, A * 2, H, W) to (N, HWA, 2).
            N, _, H, W = handside_pred.shape
            handside_pred = handside_pred.view(N, -1, 2, H, W)
            handside_pred = handside_pred.permute(0, 3, 4, 1, 2)
            handside_pred = handside_pred.reshape(N, -1, 2)  # Size=(N, HWA, 2)

            all_hand_lr.append(handside_pred)

            if self.ext:

                dxdymagnitude_pred = nn.functional.relu(self.hand_dydx_layer(cls_tower))
                dxdymagnitude_pred_sub = 0.1 * nn.functional.normalize(dxdymagnitude_pred[:,1:], p=2, dim=1)
                dxdymagnitude_pred_norm = torch.cat([dxdymagnitude_pred[:,0].unsqueeze(1), dxdymagnitude_pred_sub], dim=1)

                contactstate_pred = self.hand_contact_state_layer(cls_tower)

                # Permute offset regression output from (N, A * 3, H, W) to (N, HWA, 3).
                N, _, H, W = dxdymagnitude_pred_norm.shape
                dxdymagnitude_pred_norm = dxdymagnitude_pred_norm.view(N, -1, 3, H, W)
                dxdymagnitude_pred_norm = dxdymagnitude_pred_norm.permute(0, 3, 4, 1, 2)
                dxdymagnitude_pred_norm = dxdymagnitude_pred_norm.reshape(N, -1, 3)  # Size=(N, HWA,)

                # Permute contact classification output from (N, A * 5, H, W) to (N, HWA, 5).
                N, _, H, W = contactstate_pred.shape
                contactstate_pred = contactstate_pred.view(N, -1, 5, H, W)
                contactstate_pred = contactstate_pred.permute(0, 3, 4, 1, 2)
                contactstate_pred = contactstate_pred.reshape(N, -1, 5)  # Size=(N, HWA, 5)

                all_hand_contact_state.append(contactstate_pred)
                all_hand_dxdy.append(dxdymagnitude_pred_norm)
                

            feature_idx = torch.full((cls_logits.shape[0], cls_logits.shape[1], 1), idx)
            all_feature_idx.append(feature_idx)

        if self.ext:
            return torch.cat(all_cls_logits, dim=1), torch.cat(all_hand_contact_state, dim=1), torch.cat(all_hand_dxdy, dim=1), torch.cat(all_hand_lr, dim=1), torch.cat(all_feature_idx, dim=1)
        else:
            return torch.cat(all_cls_logits, dim=1), None, None, torch.cat(all_hand_lr, dim=1), torch.cat(all_feature_idx, dim=1)


class FCOSRegressionHead(nn.Module):
    """
    A regression head for use in FCOS.

    Args:
        in_channels (int): number of channels of the input feature
        num_anchors (int): number of anchors to be predicted
        num_convs (Optional[int]): number of conv layer. Default: 4.
        norm_layer: Module specifying the normalization layer to use.
    """

    def __init__(
        self,
        in_channels: int,
        num_anchors: int,
        num_convs: int = 4,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ):
        super().__init__()

        if norm_layer is None:
            norm_layer = partial(nn.GroupNorm, 32)

        conv = []
        for _ in range(num_convs):
            conv.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1))
            conv.append(norm_layer(in_channels))
            conv.append(nn.ReLU())
        self.conv = nn.Sequential(*conv)

        self.bbox_reg = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=3, stride=1, padding=1)
        self.bbox_ctrness = nn.Conv2d(in_channels, num_anchors * 1, kernel_size=3, stride=1, padding=1)
        for layer in [self.bbox_reg, self.bbox_ctrness]:
            torch.nn.init.normal_(layer.weight, std=0.01)
            torch.nn.init.zeros_(layer.bias)

        for layer in self.conv.children():
            if isinstance(layer, nn.Conv2d):
                torch.nn.init.normal_(layer.weight, std=0.01)
                torch.nn.init.zeros_(layer.bias)

    def forward(self, x: List[Tensor]) -> Tuple[Tensor, Tensor]:
        all_bbox_regression = []
        all_bbox_ctrness = []

        for features in x:
            bbox_feature = self.conv(features)
            bbox_regression = nn.functional.relu(self.bbox_reg(bbox_feature))
            bbox_ctrness = self.bbox_ctrness(bbox_feature)

            # permute bbox regression output from (N, 4 * A, H, W) to (N, HWA, 4).
            N, _, H, W = bbox_regression.shape
            bbox_regression = bbox_regression.view(N, -1, 4, H, W)
            bbox_regression = bbox_regression.permute(0, 3, 4, 1, 2)
            bbox_regression = bbox_regression.reshape(N, -1, 4)  # Size=(N, HWA, 4)
            all_bbox_regression.append(bbox_regression)

            # permute bbox ctrness output from (N, 1 * A, H, W) to (N, HWA, 1).
            bbox_ctrness = bbox_ctrness.view(N, -1, 1, H, W)
            bbox_ctrness = bbox_ctrness.permute(0, 3, 4, 1, 2)
            bbox_ctrness = bbox_ctrness.reshape(N, -1, 1)
            all_bbox_ctrness.append(bbox_ctrness)

        return torch.cat(all_bbox_regression, dim=1), torch.cat(all_bbox_ctrness, dim=1)


class FCOS(nn.Module):
    """
    Implements FCOS.

    The input to the model is expected to be a list of tensors, each of shape [C, H, W], one for each
    image, and should be in 0-1 range. Different images can have different sizes.

    The behavior of the model changes depending if it is in training or evaluation mode.

    During training, the model expects both the input tensors, as well as a targets (list of dictionary),
    containing:
        - boxes (``FloatTensor[N, 4]``): the ground-truth boxes in ``[x1, y1, x2, y2]`` format, with
          ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (Int64Tensor[N]): the class label for each ground-truth box

    The model returns a Dict[Tensor] during training, containing the classification, regression
    and centerness losses.

    During inference, the model requires only the input tensors, and returns the post-processed
    predictions as a List[Dict[Tensor]], one for each input image. The fields of the Dict are as
    follows:
        - boxes (``FloatTensor[N, 4]``): the predicted boxes in ``[x1, y1, x2, y2]`` format, with
          ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (Int64Tensor[N]): the predicted labels for each image
        - scores (Tensor[N]): the scores for each prediction

    Args:
        backbone (nn.Module): the network used to compute the features for the model.
            It should contain an out_channels attribute, which indicates the number of output
            channels that each feature map has (and it should be the same for all feature maps).
            The backbone should return a single Tensor or an OrderedDict[Tensor].
        num_classes (int): number of output classes of the model (including the background).
        min_size (int): minimum size of the image to be rescaled before feeding it to the backbone
        max_size (int): maximum size of the image to be rescaled before feeding it to the backbone
        image_mean (Tuple[float, float, float]): mean values used for input normalization.
            They are generally the mean values of the dataset on which the backbone has been trained
            on
        image_std (Tuple[float, float, float]): std values used for input normalization.
            They are generally the std values of the dataset on which the backbone has been trained on
        anchor_generator (AnchorGenerator): module that generates the anchors for a set of feature
            maps. For FCOS, only set one anchor for per position of each level, the width and height equal to
            the stride of feature map, and set aspect ratio = 1.0, so the center of anchor is equivalent to the point
            in FCOS paper.
        head (nn.Module): Module run on top of the feature pyramid.
            Defaults to a module containing a classification and regression module.
        center_sampling_radius (int): radius of the "center" of a groundtruth box,
            within which all anchor points are labeled positive.
        score_thresh (float): Score threshold used for postprocessing the detections.
        nms_thresh (float): NMS threshold used for postprocessing the detections.
        detections_per_img (int): Number of best detections to keep after NMS.
        topk_candidates (int): Number of best detections to keep before NMS.
    """

    __annotations__ = {
        "box_coder": det_utils.BoxLinearCoder,
    }

    def __init__(
        self,
        num_classes: int,
        # transform parameters
        ext: bool = True,
        min_size: int = 800,
        max_size: int = 1333,
        image_mean: Optional[List[float]] = None,
        image_std: Optional[List[float]] = None,
        # Anchor parameters
        anchor_generator: Optional[AnchorGenerator] = None,
        head: Optional[nn.Module] = None,
        center_sampling_radius: float = 1.5,
        score_thresh: float = 0.2,
        nms_thresh: float = 0.6,
        detections_per_img: int = 100,
        topk_candidates: int = 1000,
    ):
        super().__init__()
        self.ext = ext

        backbone = resnet_fpn_backbone('resnet34', pretrained=True, returned_layers=[2,3,4])

        if not hasattr(backbone, "out_channels"):
            raise ValueError(
                "backbone should contain an attribute out_channels "
                "specifying the number of output channels (assumed to be the "
                "same for all the levels)"
            )
        self.backbone = backbone

        assert isinstance(anchor_generator, (AnchorGenerator, type(None)))

        if anchor_generator is None:
            anchor_sizes = ((8,), (16,), (32,),)  # equal to strides of multi-level feature map
            aspect_ratios = ((1.0,),) * len(anchor_sizes)  # set only one anchor
            anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)
        self.anchor_generator = anchor_generator
        assert self.anchor_generator.num_anchors_per_location()[0] == 1

        if head is None:
            head = FCOSHead(backbone.out_channels, anchor_generator.num_anchors_per_location()[0], num_classes, ext=self.ext)
        self.head = head

        self.box_coder = det_utils.BoxLinearCoder(normalize_by_size=True)

        if image_mean is None:
            image_mean = [0.485, 0.456, 0.406]
        if image_std is None:
            image_std = [0.229, 0.224, 0.225]
        self.transform = GeneralizedRCNNTransform(min_size, max_size, image_mean, image_std)

        self.center_sampling_radius = center_sampling_radius
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.detections_per_img = detections_per_img
        self.topk_candidates = topk_candidates

        # used only on torchscript mode
        self._has_warned = False

    @torch.jit.unused
    def eager_outputs(
        self, losses: Dict[str, Tensor], detections: List[Dict[str, Tensor]]) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]:
        if self.training:
            return losses
        return detections

    def compute_loss(
        self,
        targets: List[Dict[str, Tensor]],
        head_outputs: Dict[str, Tensor],
        anchors: List[Tensor],
        num_anchors_per_level: List[int],
    ) -> Dict[str, Tensor]:
        matched_idxs = []
        for anchors_per_image, targets_per_image in zip(anchors, targets):
            if targets_per_image["boxes"].numel() == 0:
                matched_idxs.append(
                    torch.full((anchors_per_image.size(0),), -1, dtype=torch.int64, device=anchors_per_image.device)
                )
                continue

            gt_boxes = targets_per_image["boxes"]
            gt_centers = (gt_boxes[:, :2] + gt_boxes[:, 2:]) / 2  # Nx2
            anchor_centers = (anchors_per_image[:, :2] + anchors_per_image[:, 2:]) / 2  # N
            anchor_sizes = anchors_per_image[:, 2] - anchors_per_image[:, 0]
            # center sampling: anchor point must be close enough to gt center.
            pairwise_match = (anchor_centers[:, None, :] - gt_centers[None, :, :]).abs_().max(
                dim=2
            ).values < self.center_sampling_radius * anchor_sizes[:, None]
            # compute pairwise distance between N points and M boxes
            x, y = anchor_centers.unsqueeze(dim=2).unbind(dim=1)  # (N, 1)
            x0, y0, x1, y1 = gt_boxes.unsqueeze(dim=0).unbind(dim=2)  # (1, M)
            pairwise_dist = torch.stack([x - x0, y - y0, x1 - x, y1 - y], dim=2)  # (N, M)

            # anchor point must be inside gt
            pairwise_match &= pairwise_dist.min(dim=2).values > 0

            # each anchor is only responsible for certain scale range.
            lower_bound = anchor_sizes * 4
            lower_bound[: num_anchors_per_level[0]] = 0
            upper_bound = anchor_sizes * 8
            upper_bound[-num_anchors_per_level[-1] :] = float("inf")
            pairwise_dist = pairwise_dist.max(dim=2).values
            pairwise_match &= (pairwise_dist > lower_bound[:, None]) & (pairwise_dist < upper_bound[:, None])

            # match the GT box with minimum area, if there are multiple GT matches
            gt_areas = (gt_boxes[:, 1] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])  # N
            pairwise_match = pairwise_match.to(torch.float32) * (1e8 - gt_areas[None, :])
            min_values, matched_idx = pairwise_match.max(dim=1)  # R, per-anchor match
            matched_idx[min_values < 1e-5] = -1  # unmatched anchors are assigned -1

            matched_idxs.append(matched_idx)

        return self.head.compute_loss(targets, head_outputs, anchors, matched_idxs)

    def postprocess_detections(
        self, head_outputs: Dict[str, List[Tensor]], anchors: List[List[Tensor]],num_anchors_per_level: List[int]
    ) -> List[Dict[str, Tensor]]:

        class_logits = head_outputs["cls_logits"]
        bbox_regression = head_outputs["bbox_regression"]
        box_ctrness = head_outputs["bbox_ctrness"]
        hand_lr = head_outputs["hand_lr"]
        feature_idx = head_outputs["feature_idx"]

        if self.ext:
            hand_contact_state = head_outputs["hand_contact_state"]
            hand_dxdy = head_outputs["hand_dxdy"]
        else:
            hand_contact_state_max = torch.zeros_like(box_ctrness)
            hand_dxdy = torch.zeros_like(box_ctrness)

        detections: List[Dict[str, Tensor]] = []

        pred_boxes = [
                self.box_coder.decode_single(bbox_regression_per_image, anchors_per_image).unsqueeze(0)
                for anchors_per_image, bbox_regression_per_image in zip(anchors, bbox_regression)
            ]
        
        pred_boxes = torch.cat(pred_boxes)

        scores = torch.sqrt( torch.sigmoid( class_logits) *  torch.sigmoid(box_ctrness) )
        scores_max, labels_max = torch.max(scores, dim=-1)
        masks = scores_max > 0.7

        sides = torch.sigmoid(hand_lr)
        _, sides_max = torch.max(sides, dim=-1)
        
        
        if self.ext:
            hand_contact_state = torch.sigmoid(hand_contact_state)
            _, hand_contact_state_max = torch.max(hand_contact_state, dim=-1)

        level_anchors = psum(num_anchors_per_level)

        feature_idxs = []
        for anchor in anchors:
            feature_idx = torch.zeros(anchor.shape[0])
            for i in range(1, len(level_anchors) - 1):
                feature_idx[level_anchors[i] : level_anchors[i+1]] = i
            feature_idxs.append(feature_idx.unsqueeze(0))
        feature_idxs = torch.cat(feature_idxs)


        for boxes, scores, labels, sides, hand_contact_state, hand_dxdy, feature_idx, mask in zip(
            pred_boxes, scores_max, labels_max, sides_max, hand_contact_state_max, hand_dxdy, feature_idxs, masks):
            
            boxes = boxes[mask]
            scores = scores[mask]
            labels = labels[mask]
            sides = sides[mask]
            feature_idx = feature_idx[mask]

            if self.ext:
                hand_contact_state = hand_contact_state[mask]
                hand_dxdy = hand_dxdy[mask]
            

            keep = box_ops.batched_nms(boxes, scores, labels, 0.3)
            
            if self.ext:
                detections.append(
                    {
                        "boxes": boxes[keep],
                        "scores": scores[keep],
                        "labels": labels[keep],
                        "dxdymags": hand_dxdy[keep],
                        "contacts": hand_contact_state[keep].reshape(-1),
                        "sides": sides[keep].reshape(-1),
                    }
                )
            else:
                detections.append(
                    {
                        "boxes": boxes[keep],
                        "scores": scores[keep],
                        "labels": labels[keep],
                        "sides": sides[keep].reshape(-1),
                        "feature_idx": feature_idx[keep].reshape(-1),
                    }
                )

        return detections

    def postprocess(self, result: List[Dict[str, Tensor]],
                    image_shapes: List[Tuple[int, int]],
                    original_image_sizes: List[Tuple[int, int]]
                    ) -> List[Dict[str, Tensor]]:
        for i, (pred, im_s, o_im_s) in enumerate(zip(result, image_shapes, original_image_sizes)):
            boxes = pred["boxes"]
            boxes = resize_boxes(boxes, im_s, o_im_s)
            result[i]["boxes"] = boxes
        return result
    
    def postprocess_single(self, boxes, image_shape, original_image_size):
        boxes = resize_boxes(boxes, image_shape, original_image_size)
        return boxes

    def forward(
        self,
        images: List[Tensor],
        targets: Optional[List[Dict[str, Tensor]]] = None,
    ) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]:
        """
        Args:
            images (list[Tensor]): images to be processed
            targets (list[Dict[Tensor]]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).
        """
        if self.training:
            if targets is None:
                raise ValueError("In training mode, targets should be passed")
            for target in targets:
                boxes = target["boxes"]
                if isinstance(boxes, torch.Tensor):
                    if len(boxes.shape) != 2 or boxes.shape[-1] != 4:
                        raise ValueError(f"Expected target boxes to be a tensor of shape [N, 4], got {boxes.shape}.")
                else:
                    raise ValueError(f"Expected target boxes to be of type Tensor, got {type(boxes)}.")

        original_image_sizes: List[Tuple[int, int]] = []
        for img in images:
            val = img.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))

        # transform the input
        images, targets = self.transform(images, targets)

        # Check for degenerate boxes
        valid_idx = []
        if targets is not None:
            for target_idx, target in enumerate(targets):
                boxes = target["boxes"]
                degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
                if degenerate_boxes.any():
                    # print the first degenerate box
                    bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                    valid_idx_degen = torch.where(~degenerate_boxes.any(dim=1))[0]
                    valid_idx_degen.to(boxes.device)
                    valid_idx.append(valid_idx_degen)
                    degen_bb: List[float] = boxes[bb_idx].tolist()
                    print(
                        "All bounding boxes should have positive height and width."
                        f" Found first invalid box {degen_bb} for box {bb_idx} in image_id {target['image_id'][0]}."
                    )
                else:
                    valid_idx.append(torch.arange(boxes.shape[0], device=boxes.device))

        if targets is not None and len(valid_idx) > 0:
            for target_idx, _ in enumerate(targets):
                targets[target_idx]["boxes"] = torch.index_select(targets[target_idx]["boxes"], 0, valid_idx[target_idx])
                

        # get the features from the backbone
        features = self.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])

        features = list(features.values())
        features = features[:-1] # remove max pooling layer

        # compute the fcos heads outputs using the features
        head_outputs = self.head(features)

        # create the set of anchors
        anchors = self.anchor_generator(images, features)
        # recover level sizes
        num_anchors_per_level = [x.size(2) * x.size(3) for x in features]

        losses = {}
        detections = self.postprocess_detections(head_outputs, anchors, num_anchors_per_level)
        if self.training:
            assert targets is not None
            # compute the losses
            losses = self.compute_loss(targets, head_outputs, anchors, num_anchors_per_level)
        else:
            # compute the detections
            detections = self.postprocess(detections, images.image_sizes, original_image_sizes)

        if torch.jit.is_scripting():
            if not self._has_warned:
                warnings.warn("FCOS always returns a (Losses, Detections) tuple in scripting")
                self._has_warned = True
            return losses, detections
        return self.eager_outputs(losses, detections)


def resize_boxes(boxes: Tensor, original_size: List[int], new_size: List[int]) -> Tensor:
    ratios = [
        torch.tensor(s, dtype=torch.float32, device=boxes.device) /
        torch.tensor(s_orig, dtype=torch.float32, device=boxes.device)
        for s, s_orig in zip(new_size, original_size)
    ]
    ratio_height, ratio_width = ratios
    xmin, ymin, xmax, ymax = boxes.unbind(1)

    xmin = xmin * ratio_width
    xmax = xmax * ratio_width
    ymin = ymin * ratio_height
    ymax = ymax * ratio_height
    return torch.stack((xmin, ymin, xmax, ymax), dim=1)


def psum(a):
	psum=[0]
	for i in a:
		psum.append(psum[-1]+i) # psum[-1] is the last element in the list
	return psum


# class FCOSLightningModel(pl.LightningModule):
#     def __init__(
#         self,
#         num_classes: int = 3,
#         # transform parameters
#         ext: bool = True,
#         min_size: int = 800,
#         max_size: int = 1333,
#         image_mean: Optional[List[float]] = None,
#         image_std: Optional[List[float]] = None,
#         center_sampling_radius: float = 1.5,
#         score_thresh: float = 0.2,
#         nms_thresh: float = 0.6,
#         detections_per_img: int = 100,
#         topk_candidates: int = 1000,
#     ):
#         super().__init__()
#         self.fcos = FCOS(
#             num_classes=num_classes,
#             ext=ext,
#             min_size=min_size,
#             max_size=max_size,
#             image_mean=image_mean,
#             image_std=image_std,
#             center_sampling_radius=center_sampling_radius,
#             score_thresh=score_thresh,
#             nms_thresh=nms_thresh,
#             detections_per_img=detections_per_img,
#             topk_candidates=topk_candidates,
#         )
