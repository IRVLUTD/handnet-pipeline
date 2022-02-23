from copy import deepcopy

import torch
from torch import nn
import torch.nn.functional as torch_f

from mano_train.networks.bases import resnet
from mano_train.networks.branches.manobranch import ManoBranch, ManoLoss
from mano_train.networks.branches.absolutebranch import AbsoluteBranch
from datasets3d.queries import TransQueries, BaseQueries

def get_base_net(cls):
    return cls.module.base_net

class HandNet(nn.Module):
    def __init__(
        self,
        in_channels,
        fc_dropout=0,
        mano_adapt_skeleton=False,
        mano_neurons=[512],
        mano_comps=45,
        mano_use_shape=True,
        mano_use_pca=True,
        mano_root="misc/mano/models",
        mano_use_joints2d=True,
        mano_use_trans=False,
    ):
        """
        Args:
            in_channels: channels of roi after convfchead
            mano_root (path): dir containing mano pickle files
            mano_neurons: number of neurons in each layer of base mano decoder
            mano_use_pca: predict pca parameters directly instead of rotation
                angles
            mano_comps (int): number of principal components to use if
                mano_use_pca
        """
        super().__init__()
        mano_base_neurons = [in_channels] + mano_neurons

        self.scaletrans_branch = AbsoluteBranch(
                base_neurons=[in_channels, int(in_channels / 2)],
                out_dim=3,
        )
        self.mano_use_joints2d = mano_use_joints2d

        self.mano_adapt_skeleton = mano_adapt_skeleton
        self.mano_branch = ManoBranch(
            ncomps=mano_comps,
            base_neurons=mano_base_neurons,
            adapt_skeleton=mano_adapt_skeleton,
            dropout=fc_dropout,
            use_trans=mano_use_trans,
            mano_root=mano_root,
            use_shape=mano_use_shape,
            use_pca=mano_use_pca,
        )
        self.cuda()

    def forward(
        self, sample, return_features=False
    ):
        device = torch.device("cuda")
        features = sample[BaseQueries.features].to(device)
        sample[TransQueries.joints3d].to(device)
        results = {}
        if return_features:
            results["img_features"] = features

        if (
            (
                TransQueries.joints3d in sample.keys()
                or TransQueries.verts3d in sample.keys()
                or (
                    TransQueries.joints2d in sample.keys()
                    and TransQueries.camintrs in sample.keys()
                )
            )
            and BaseQueries.sides in sample.keys()
        ):
            mano_results = self.mano_branch(
                features,
                sides=sample[BaseQueries.sides],
                use_stereoshape=False,
            )

            for key, result in mano_results.items():
                results[key] = result

            if self.mano_use_joints2d:
                scaletrans = self.scaletrans_branch(features)
                trans = scaletrans[:, 1:]
                # Abs to make sure no inversion in scale
                scale = torch.abs(scaletrans[:, :1])

                # Trans is multiplied by 100 to make scale and trans updates
                # of same magnitude after 2d joints supervision
                # (100 is ~ the scale of the 2D joint coordinate values)
                proj_joints2d = mano_results["joints"][
                    :, :, :2
                ] * scale.unsqueeze(1) + 100 * trans.unsqueeze(1)
                results["joints2d"] = proj_joints2d

        return results