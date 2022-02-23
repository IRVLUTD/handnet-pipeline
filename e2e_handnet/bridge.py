from socket import TCP_WINDOW_CLAMP
from turtle import forward
from torch import nn, Tensor
import torch

import torchvision
from torchvision.ops import roi_align

from typing import Optional, List, Dict, Tuple, Union
from fpn_utils.faster_rcnn_fpn import TwoMLPHead
from datasets3d.queries import TransQueries, BaseQueries

class E2EBridge(nn.Module):
    def __init__(
        self,
        out_channels: int = 256,
    ):

        super().__init__()

        self.roi_align = MultiScaleRoIAlign(
                featmap_levels=['0', '1', '2'],
                output_size=7,
                sampling_ratio=2)

        resolution = self.roi_align.output_size[0]
        representation_size = 1024
        self.head_3d = TwoMLPHead(
            out_channels * resolution ** 2,
            representation_size,
        )

    def forward(
        self, sample, features, boxes, image_shapes, sides
    ):
        aligned_features, batch_idx, level_idx = self.roi_align(features, boxes, image_shapes)
        aligned_features = self.head_3d(aligned_features)
        
        target = {
            'idx': sample['idx'][batch_idx[:]],
            'dexycb_id': sample['dexycb_id'][batch_idx[:]],
            TransQueries.verts3d: sample[TransQueries.verts3d][batch_idx[:]],
            TransQueries.joints3d: sample[TransQueries.joints3d][batch_idx[:]],
            TransQueries.joints2d: sample[TransQueries.joints2d][batch_idx[:]],
            BaseQueries.sides: sample[BaseQueries.sides][batch_idx[:]],
            #BaseQueries.sides: torch.cat(sides),
            BaseQueries.features: aligned_features
        }

        return target, batch_idx, level_idx



class MultiScaleRoIAlign(nn.Module):
    """
    Multi-scale RoIAlign pooling

    Because the feature level for each bounding box is specified by FCOS, we can avoid canonical heuristics defined in eq 1. of the `Feature Pyramid Network paper <https://arxiv.org/abs/1612.03144>`_.

    Args:
        featmap_levels (List[str]): the names of the feature maps that will be used
            for the pooling.
        output_size (List[Tuple[int, int]] or List[int]): output size for the pooled region
        sampling_ratio (int): sampling ratio for ROIAlign
    """

    __annotations__ = {
        'scales': Optional[List[float]],
    }

    def __init__(
        self,
        featmap_levels: List[str],
        output_size: Union[int, Tuple[int], List[int]],
        sampling_ratio: int,
    ):
        super(MultiScaleRoIAlign, self).__init__()
        if isinstance(output_size, int):
            output_size = (output_size, output_size)
        self.featmap_levels = featmap_levels
        self.sampling_ratio = sampling_ratio
        self.output_size = tuple(output_size)
        self.scales = None

    def convert_to_roi_format(self, boxes: List[Tensor]) -> Tensor:
        concat_boxes = torch.cat(boxes, dim=0)
        device, dtype = concat_boxes.device, concat_boxes.dtype
        ids = torch.cat(
            [
                torch.full_like(b[:, :1], i, dtype=dtype, layout=torch.strided, device=device)
                for i, b in enumerate(boxes)
            ],
            dim=0,
        )
        # rois = torch.cat([ids, concat_boxes], dim=1)
        # boxes = boxes.reshape(boxes.shape[0], -1, 4)
        # device, dtype = boxes.device, boxes.dtype
        # ids = torch.cat(
        #     [
        #         torch.full_like(b[:, :1], i, dtype=dtype, layout=torch.strided, device=device)
        #         for i, b in enumerate(boxes)
        #     ],
        #     dim=0,
        # )   
        rois = torch.cat([ids, concat_boxes], dim=1)
        return rois

    def infer_scale(self, feature: Tensor, original_size: List[int]) -> float:
        # assumption: the scale is of the form 2 ** (-k), with k integer
        size = feature.shape[-2:]
        possible_scales: List[float] = []
        for s1, s2 in zip(size, original_size):
            approx_scale = float(s1) / float(s2)
            scale = 2 ** float(torch.tensor(approx_scale).log2().round())
            possible_scales.append(scale)
        assert possible_scales[0] == possible_scales[1]
        return possible_scales[0]

    def setup_scales(
        self,
        features: List[Tensor],
        image_shapes: List[Tuple[int, int]],
    ) -> None:
        assert len(image_shapes) != 0
        max_x = 0
        max_y = 0
        for shape in image_shapes:
            max_x = max(shape[0], max_x)
            max_y = max(shape[1], max_y)
        original_input_shape = (max_x, max_y)

        self.scales = [self.infer_scale(feat, original_input_shape) for feat in features]

    def forward(
        self,
        x: Dict[str, Tensor],
        boxes,
        image_shapes: List[Tuple[int, int]],
    ) -> Tensor:
        """
        Args:
            x (OrderedDict[Tensor]): feature maps for each level. They are assumed to have
                all the same number of channels, but they can have different sizes.
            boxes List[List[Tensor[N, 4]]]: boxes for each image in each level to be used to perform the pooling operation, in
                (x1, y1, x2, y2) format and in the image reference size, not the feature map
                reference. The coordinate must satisfy ``0 <= x1 < x2`` and ``0 <= y1 < y2``.
            image_shapes (List[Tuple[height, width]]): the sizes of each image before they
                have been fed to a CNN to obtain feature maps. This allows us to infer the
                scale factor for each one of the levels to be pooled.
        Returns:
            result (Tensor)
        """

        x_filtered = []
        for k, v in x.items():
            if k in self.featmap_levels:
                x_filtered.append(v)
        num_levels = len(x_filtered)

        # F = number of feature maps
        # boxes | F * N * N_H_i_f * 4
        rois = []
        num_rois = []
        for boxes_per_level in boxes:
            rois_per_level = self.convert_to_roi_format(boxes_per_level)
            rois.append(rois_per_level)
            num_rois.append(len(rois_per_level))

        # rois | F * (N_H_f * 4)
        # num_rois | F = number of rois per level
        
        if self.scales is None:
            self.setup_scales(x_filtered, image_shapes)

        scales = self.scales
        assert scales is not None

        if num_levels == 1:
            return roi_align(
                x_filtered[0], rois,
                output_size=self.output_size,
                spatial_scale=scales[0],
                sampling_ratio=self.sampling_ratio
            )

        num_channels = x_filtered[0].shape[1]

        dtype, device = x_filtered[0].dtype, x_filtered[0].device

        result = []
        batch_idx = []
        level_idx = []
        for i in range(num_levels):
            result.append(torch.zeros((num_rois[i], num_channels,) + self.output_size,
                dtype=dtype,
                device=device,))
            batch_idx.append(torch.zeros(
                (num_rois[i]),
                dtype=torch.int64,
                device=device,
            ))
            level_idx.append(torch.full_like( 
                batch_idx[i], 
                i, 
                dtype=torch.int64, 
                device=device
            ))

        tracing_results = []
        for level, (per_level_feature, scale, per_level_rois) in enumerate(zip(x_filtered, scales, rois)):

            result_idx_in_level = roi_align(
                per_level_feature, per_level_rois,
                output_size=self.output_size,
                spatial_scale=scale, sampling_ratio=self.sampling_ratio)

                # result and result_idx_in_level's dtypes are based on dtypes of different
                # elements in x_filtered.  x_filtered contains tensors output by different
                # layers.  When autocast is active, it may choose different dtypes for
                # different layers' outputs.  Therefore, we defensively match result's dtype
                # before copying elements from result_idx_in_level in the following op.
                # We need to cast manually (can't rely on autocast to cast for us) because
                # the op acts on result in-place, and autocast only affects out-of-place ops.
            result[level] = result_idx_in_level.to(dtype)
            batch_idx[level] = per_level_rois[:, 0].to(torch.int64)

        return torch.cat(result, dim=0), torch.cat(batch_idx, dim=0), torch.cat(level_idx, dim=0) # final result | (sum(N_H ), ...)

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}(featmap_names={self.featmap_levels}, "
                f"output_size={self.output_size}, sampling_ratio={self.sampling_ratio})")
