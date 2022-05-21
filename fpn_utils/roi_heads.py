import torch
import torchvision

import torch.nn.functional as F
from torch import Tensor

from torchvision.ops import boxes as box_ops

from torchvision.ops import roi_align

from torchvision.models.detection import _utils as det_utils

from torch.jit.annotations import Optional, List, Dict, Tuple


def fastrcnn_loss(box_logits, class_logits, box_regression, labels, box_info, regression_targets, ext=True):
    # type: (List[Tensor], Tensor, Tensor, List[Tensor], List[Tensor], List[Tensor], bool) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]
    """
    Computes the loss for Faster R-CNN.

    Arguments:
        class_logits (Tensor)
        box_regression (Tensor)
        labels (list[BoxList])
        regression_targets (Tensor)
        ext (bool)

    Returns:
        classification_loss (Tensor)
        box_loss (Tensor)
    """

    labels = torch.cat(labels, dim=0)
    regression_targets = torch.cat(regression_targets, dim=0)
    box_info = torch.cat(box_info, dim=0)

    classification_loss = F.cross_entropy(class_logits, labels)

    # get indices that correspond to the regression targets for
    # the corresponding ground truth labels, to be used with
    # advanced indexing
    sampled_pos_inds_subset = torch.where(labels > 0)[0]
    labels_pos = labels[sampled_pos_inds_subset]
    N, num_classes = class_logits.shape
    box_regression = box_regression.reshape(N, -1, 4)

    if ext:
        handside_logits = box_logits[0].reshape(N, -1, 1)
        dxdymag_regression = box_logits[1].reshape(N, -1, 3)
        contact_regression = box_logits[2].reshape(N, -1, 5)

    box_loss = F.smooth_l1_loss(
        box_regression[sampled_pos_inds_subset, labels_pos],
        regression_targets[sampled_pos_inds_subset],
        beta=1 / 9,
        reduction="sum",
    )
    box_loss = box_loss / labels.numel()

    if ext:
        handside_loss = 0.1 * F.binary_cross_entropy_with_logits(
            handside_logits[sampled_pos_inds_subset, labels_pos].squeeze(),
            box_info[sampled_pos_inds_subset, 1].squeeze()
        )

        dxdymag_loss = 0.1 * F.mse_loss(
            dxdymag_regression[sampled_pos_inds_subset, labels_pos],
            box_info[sampled_pos_inds_subset, 2:]
        )

        contact_loss = 0.1 * F.cross_entropy(
            contact_regression[sampled_pos_inds_subset, labels_pos],
            box_info[sampled_pos_inds_subset, 0].long()
        )
    else:
        handside_loss = 0
        dxdymag_loss = 0
        contact_loss = 0

    return handside_loss, dxdymag_loss, contact_loss, classification_loss, box_loss


class RoIHeads(torch.nn.Module):
    __annotations__ = {
        'box_coder': det_utils.BoxCoder,
        'proposal_matcher': det_utils.Matcher,
        'fg_bg_sampler': det_utils.BalancedPositiveNegativeSampler,
    }

    def __init__(self,
                 box_roi_pool,
                 box_head,
                 box_predictor,
                 # Faster R-CNN training
                 fg_iou_thresh, bg_iou_thresh,
                 batch_size_per_image, positive_fraction,
                 bbox_reg_weights,
                 # Faster R-CNN inference
                 score_thresh,
                 nms_thresh,
                 detections_per_img,
                 ext=True
                 ):
        super(RoIHeads, self).__init__()

        self.ext = ext

        self.box_similarity = box_ops.box_iou
        # assign ground-truth boxes for each proposal
        self.proposal_matcher = det_utils.Matcher(
            fg_iou_thresh,
            bg_iou_thresh,
            allow_low_quality_matches=False)

        self.fg_bg_sampler = det_utils.BalancedPositiveNegativeSampler(
            batch_size_per_image,
            positive_fraction)

        if bbox_reg_weights is None:
            bbox_reg_weights = (10., 10., 5., 5.)
        self.box_coder = det_utils.BoxCoder(bbox_reg_weights)

        self.box_roi_pool = box_roi_pool
        self.box_head = box_head
        self.box_predictor = box_predictor

        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.detections_per_img = detections_per_img

    def assign_targets_to_proposals(self, proposals, gt_boxes, gt_labels, gt_box_info):
        # type: (List[Tensor], List[Tensor], List[Tensor], List[Tensor]) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]
        matched_idxs = []
        labels = []
        box_info = []
        for proposals_in_image, gt_boxes_in_image, gt_labels_in_image, gt_box_info_in_image in zip(proposals, gt_boxes, gt_labels, gt_box_info):

            if gt_boxes_in_image.numel() == 0:
                # Background image
                device = proposals_in_image.device
                clamped_matched_idxs_in_image = torch.zeros(
                    (proposals_in_image.shape[0],), dtype=torch.int64, device=device
                )
                labels_in_image = torch.zeros(
                    (proposals_in_image.shape[0],), dtype=torch.int64, device=device
                )
                box_info_in_image = torch.zeros(
                    (proposals_in_image.shape[0],), dtype=torch.float32, device=device
                )
            else:
                match_quality_matrix = box_ops.box_iou(gt_boxes_in_image, proposals_in_image)
                matched_idxs_in_image = self.proposal_matcher(match_quality_matrix)

                clamped_matched_idxs_in_image = matched_idxs_in_image.clamp(min=0)

                labels_in_image = gt_labels_in_image[clamped_matched_idxs_in_image]
                labels_in_image = labels_in_image.to(dtype=torch.int64)

                box_info_in_image = gt_box_info_in_image[clamped_matched_idxs_in_image]
                box_info_in_image = box_info_in_image.to(dtype=torch.float32)

                # Label background (below the low threshold)
                bg_inds = matched_idxs_in_image == self.proposal_matcher.BELOW_LOW_THRESHOLD
                labels_in_image[bg_inds] = 0
                box_info_in_image[bg_inds] = 0

                # Label ignore proposals (between low and high thresholds)
                ignore_inds = matched_idxs_in_image == self.proposal_matcher.BETWEEN_THRESHOLDS
                labels_in_image[ignore_inds] = -1  # -1 is ignored by sampler
                box_info_in_image[ignore_inds] = 0

            matched_idxs.append(clamped_matched_idxs_in_image)
            labels.append(labels_in_image)
            box_info.append(box_info_in_image)
        return matched_idxs, labels, box_info

    def subsample(self, labels):
        # type: (List[Tensor]) -> List[Tensor]
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
        sampled_inds = []
        for img_idx, (pos_inds_img, neg_inds_img) in enumerate(
            zip(sampled_pos_inds, sampled_neg_inds)
        ):
            img_sampled_inds = torch.where(pos_inds_img | neg_inds_img)[0]
            sampled_inds.append(img_sampled_inds)
        return sampled_inds

    def add_gt_proposals(self, proposals, gt_boxes):
        # type: (List[Tensor], List[Tensor]) -> List[Tensor]
        proposals = [
            torch.cat((proposal, gt_box))
            for proposal, gt_box in zip(proposals, gt_boxes)
        ]

        return proposals

    def check_targets(self, targets):
        # type: (Optional[List[Dict[str, Tensor]]]) -> None
        assert targets is not None
        assert all(["boxes" in t for t in targets])
        assert all(["labels" in t for t in targets])
        assert all(["box_info" in t for t in targets]) # for extension regression


    def select_training_samples(self,
                                proposals,  # type: List[Tensor]
                                targets     # type: Optional[List[Dict[str, Tensor]]]
                                ):
        # type: (...) -> Tuple[List[Tensor], List[Tensor], List[Tensor], List[Tensor], List[Tensor]]
        self.check_targets(targets)
        assert targets is not None
        dtype = proposals[0].dtype
        device = proposals[0].device

        gt_boxes = [t["boxes"].to(dtype) for t in targets]
        gt_labels = [t["labels"] for t in targets]
        gt_box_info = [t["box_info"] for t in targets]

        # append ground-truth bboxes to propos
        proposals = self.add_gt_proposals(proposals, gt_boxes)

        # get matching gt indices for each proposal
        matched_idxs, labels, box_info = self.assign_targets_to_proposals(proposals, gt_boxes, gt_labels, gt_box_info)
        # sample a fixed proportion of positive-negative proposals
        sampled_inds = self.subsample(labels)
        matched_gt_boxes = []
        num_images = len(proposals)
        for img_id in range(num_images):
            img_sampled_inds = sampled_inds[img_id]
            proposals[img_id] = proposals[img_id][img_sampled_inds]
            labels[img_id] = labels[img_id][img_sampled_inds]
            box_info[img_id] = box_info[img_id][img_sampled_inds]
            matched_idxs[img_id] = matched_idxs[img_id][img_sampled_inds]

            gt_boxes_in_image = gt_boxes[img_id]
            if gt_boxes_in_image.numel() == 0:
                gt_boxes_in_image = torch.zeros((1, 4), dtype=dtype, device=device)
            matched_gt_boxes.append(gt_boxes_in_image[matched_idxs[img_id]])

        regression_targets = self.box_coder.encode(matched_gt_boxes, proposals)
        return proposals, matched_idxs, labels, box_info, regression_targets

    def postprocess_detections(self,
                               class_logits,    # type: Tensor
                               box_logits,      # type: List[Tensor]
                               box_regression,  # type: Tensor
                               proposals,       # type: List[Tensor]
                               image_shapes     # type: List[Tuple[int, int]]
                               ):
        # type: (...) -> Tuple[List[Tensor], List[Tensor], List[Tensor], List[Tensor], List[Tensor], List[Tensor]]
        device = class_logits.device
        num_classes = class_logits.shape[-1]

        boxes_per_image = [boxes_in_image.shape[0] for boxes_in_image in proposals]
        pred_boxes = self.box_coder.decode(box_regression, proposals)

        pred_scores = F.softmax(class_logits, -1)

        # outputs from extensions
        if self.ext:
            pred_sides = torch.sigmoid(box_logits[0].squeeze()) > 0.5
            pred_sides = pred_sides.to(dtype=torch.float32)

            N = class_logits.shape[0]
            pred_dxdymags = box_logits[1].reshape(N, -1, 3)
            pred_contacts = box_logits[2].reshape(N, -1, 5)
            _, pred_contacts = torch.max(pred_contacts, 2)

        pred_boxes_list = pred_boxes.split(boxes_per_image, 0)
        pred_scores_list = pred_scores.split(boxes_per_image, 0)

        if self.ext:
            pred_sides_list = pred_sides.split(boxes_per_image, 0)
            pred_dxdymags_list = pred_dxdymags.split(boxes_per_image, 0)
            pred_contacts_list = pred_contacts.split(boxes_per_image, 0)
        else:
            pred_sides_list = torch.zeros_like(class_logits)
            pred_dxdymags_list = torch.zeros_like(class_logits)
            pred_contacts_list = torch.zeros_like(class_logits)

        if self.ext:
            all_sides = []
            all_dxdymags = []
            all_contacts = []
            all_boxes = []
        all_scores = []
        all_labels = []
        for boxes, scores, sides, dxdymags, contacts, image_shape in zip(pred_boxes_list, pred_scores_list, pred_sides_list, pred_dxdymags_list, pred_contacts_list, image_shapes):
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)

            # create labels for each prediction
            labels = torch.arange(num_classes, device=device)
            labels = labels.view(1, -1).expand_as(scores)

            # remove predictions with the background label
            boxes = boxes[:, 1:]
            scores = scores[:, 1:]
            labels = labels[:, 1:]

            if self.ext:
                sides = sides[:, 1:]
                dxdymags = dxdymags[:, 1:]
                contacts = contacts[:, 1:]

            # batch everything, by making every class prediction be a separate instance
            boxes = boxes.reshape(-1, 4)
            scores = scores.reshape(-1)
            labels = labels.reshape(-1)

            if self.ext:
                sides = sides.reshape(-1)
                dxdymags = dxdymags.reshape(-1, 3)
                contacts = contacts.reshape(-1)
            

            # remove low scoring boxes
            inds = torch.where(scores > self.score_thresh)[0]
            if self.ext:
                boxes, scores, labels, \
                    sides, dxdymags, contacts = boxes[inds], scores[inds], labels[inds], \
                                                sides[inds], dxdymags[inds], contacts[inds]
            else:
                boxes, scores, labels = boxes[inds], scores[inds], labels[inds]

            # remove empty boxes
            keep = box_ops.remove_small_boxes(boxes, min_size=1e-2)
            if self.ext:
                boxes, scores, labels, \
                    sides, dxdymags, contacts = boxes[keep], scores[keep], labels[keep], \
                                                sides[keep], dxdymags[keep], contacts[keep]
            else:
                boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            # non-maximum suppression, independently done per class
            keep = box_ops.batched_nms(boxes, scores, labels, self.nms_thresh)
            # keep only topk scoring predictions
            keep = keep[:self.detections_per_img]
            if self.ext:
                boxes, scores, labels, \
                    sides, dxdymags, contacts = boxes[keep], scores[keep], labels[keep], \
                                                sides[keep], dxdymags[keep], contacts[keep]
            else:
                boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)

            if self.ext:
                all_sides.append(sides)
                all_dxdymags.append(dxdymags)
                all_contacts.append(contacts)

        if self.ext:
            return all_sides, all_dxdymags, all_contacts, all_boxes, all_scores, all_labels
        else:
            return all_boxes, all_scores, all_labels, None, None, None

    def forward(self,
                features,      # type: Dict[str, Tensor]
                proposals,     # type: List[Tensor]
                image_shapes,  # type: List[Tuple[int, int]]
                targets=None   # type: Optional[List[Dict[str, Tensor]]]
                ):
        # type: (...) -> Tuple[List[Dict[str, Tensor]], Dict[str, Tensor]]
        """
        Arguments:
            features (List[Tensor])
            proposals (List[Tensor[N, 4]])
            image_shapes (List[Tuple[H, W]])
            targets (List[Dict])
        """
        if targets is not None:
            for t in targets:
                floating_point_types = (torch.float, torch.double, torch.half)
                assert t["boxes"].dtype in floating_point_types, 'target boxes must of float type'
                assert t["labels"].dtype == torch.int64, 'target labels must of int64 type'

        if self.training:
            proposals, matched_idxs, labels, box_info, regression_targets = self.select_training_samples(proposals, targets)
        else:
            labels = None
            regression_targets = None
            matched_idxs = None

        box_features = self.box_roi_pool(features, proposals, image_shapes)
        box_features = self.box_head(box_features)
        box_logits, class_logits, box_regression = self.box_predictor(box_features)

        result = torch.jit.annotate(List[Dict[str, torch.Tensor]], [])
        losses = {}
        if self.training:
            assert labels is not None and regression_targets is not None
            loss_handside, loss_dxdymag, loss_contact, loss_classifier, loss_box_reg = fastrcnn_loss(
                box_logits, class_logits, box_regression, labels, box_info, regression_targets, ext=self.ext)

            if self.ext:
                losses = {
                    "loss_classifier": loss_classifier,
                    "loss_box_reg": loss_box_reg, 
                    "loss_handside": loss_handside,
                    "loss_dxdymag": loss_dxdymag,
                    "loss_contact": loss_contact
                }
            else:
                losses = {
                    "loss_classifier": loss_classifier,
                    "loss_box_reg": loss_box_reg,
                }
        else:
            sides, dxdymags, contacts, boxes, scores, labels = self.postprocess_detections(class_logits, box_logits, box_regression, proposals, image_shapes)
            num_images = len(boxes)
            for i in range(num_images):
                if self.ext:
                    result.append(
                        {
                            "boxes": boxes[i],
                            "labels": labels[i],
                            "scores": scores[i],
                            "sides": sides[i],
                            "dxdymags": dxdymags[i],
                            "contacts": contacts[i]
                        }
                    )
                else:
                    result.append(
                        {
                            "boxes": boxes[i],
                            "labels": labels[i],
                            "scores": scores[i]
                        }
                    )

        return result, losses
