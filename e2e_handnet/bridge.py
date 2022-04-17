from socket import TCP_WINDOW_CLAMP
from turtle import forward
from cv2 import RECURS_FILTER
from torch import nn, Tensor
import torch

import torchvision
from torchvision.ops import roi_align

from typing import Optional, List, Dict, Tuple, Union
from e2e_handnet.deconv import Deconv
from fpn_utils.faster_rcnn_fpn import TwoMLPHead
from datasets3d.queries import TransQueries, BaseQueries
from e2e_handnet.deconv import Conv

class E2EBridge(nn.Module):
    def __init__(
        self,
        out_channels: int = 8,
        output_size: int = 8,
    ):

        super().__init__()

        self.roi_align = MultiScaleRoIAlign(
                output_size=output_size,
                sampling_ratio=2)

        representation_size = 1024
        self.deconv = Deconv(downsample=2)
        self.conv1 = Conv(256, out_channels)
        self.head_3d = TwoMLPHead(
            out_channels * 64 ** 2,
            representation_size,
        )
        # self.head_3d = TwoMLPHead(
        #     256 * output_size ** 2,
        #     representation_size,
        # )

    def forward(
        self, sample, features, boxes, image_shapes, image_mask, sides=None, test=False
    ):
        aligned_features = self.roi_align(features, boxes, image_shapes)
        aligned_features = self.deconv(aligned_features)
        aligned_features = self.conv1(aligned_features)
        aligned_features = self.head_3d(aligned_features)
        
        target = {
            'idx': sample['idx'][image_mask],
            'dexycb_id': sample['dexycb_id'][image_mask],
            TransQueries.verts3d: sample[TransQueries.verts3d][image_mask],
            TransQueries.joints3d: sample[TransQueries.joints3d][image_mask],
            TransQueries.joints2d: sample[TransQueries.joints2d][image_mask],
            BaseQueries.sides: torch.cat(sides) if test else sample[BaseQueries.sides][image_mask].flatten(),
            BaseQueries.features: aligned_features
        }

        return target



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
        output_size: Union[int, Tuple[int], List[int]],
        sampling_ratio: int,
    ):
        super(MultiScaleRoIAlign, self).__init__()
        if isinstance(output_size, int):
            output_size = (output_size, output_size)
        self.sampling_ratio = sampling_ratio
        self.output_size = tuple(output_size)
        self.scales = None

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
        feature: Tensor,
        image_shapes: List[Tuple[int, int]],
    ) -> None:
        assert len(image_shapes) != 0
        max_x = 0
        max_y = 0
        for shape in image_shapes:
            max_x = max(shape[0], max_x)
            max_y = max(shape[1], max_y)
        original_input_shape = (max_x, max_y)

        self.scales = self.infer_scale(feature, original_input_shape)

    def forward(
        self,
        x: Dict[str, Tensor],
        boxes,
        image_shapes: List[Tuple[int, int]],
    ) -> Tensor:
        """
        Args:
            x [Tensor]: feature maps for each level. They are assumed to have
                all the same number of channels, but they can have different sizes.
            boxes Tensor[N, 4]: boxes for each image in each level to be used to perform the pooling operation, in
                (x1, y1, x2, y2) format and in the image reference size, not the feature map
                reference. The coordinate must satisfy ``0 <= x1 < x2`` and ``0 <= y1 < y2``.
            image_shapes (List[Tuple[height, width]]): the sizes of each image before they
                have been fed to a CNN to obtain feature maps. This allows us to infer the
                scale factor for each one of the levels to be pooled.
        Returns:
            result (Tensor)
        """

        # F = number of feature maps
        # boxes | F * N * N_H_i_f * 4
        rois = list(boxes.reshape(-1, 1, 4))

        # rois | F * (N_H_f * 4)
        # num_rois | F = number of rois per level
        
        if self.scales is None:
            self.setup_scales(x, image_shapes)

        scales = self.scales
        assert scales is not None

        dtype, device = x[0].dtype, x[0].device

        result = roi_align(
        x, rois,
        output_size=self.output_size,
        spatial_scale=scales[0], sampling_ratio=self.sampling_ratio)

                # result and result_idx_in_level's dtypes are based on dtypes of different
                # elements in x_filtered.  x_filtered contains tensors output by different
                # layers.  When autocast is active, it may choose different dtypes for
                # different layers' outputs.  Therefore, we defensively match result's dtype
                # before copying elements from result_idx_in_level in the following op.
                # We need to cast manually (can't rely on autocast to cast for us) because
                # the op acts on result in-place, and autocast only affects out-of-place ops.
        result = result.to(dtype).to(device)

        return result

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}(featmap_names={self.featmap_levels}, "
                f"output_size={self.output_size}, sampling_ratio={self.sampling_ratio})")
