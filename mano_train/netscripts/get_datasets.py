import warnings

import numpy as np
from torch.utils.data import Subset

from datasets3d import dexycb
from datasets3d.handataset import HandDataset
from datasets3d.queries import TransQueries, BaseQueries


def get_dataset(
    dat_name,
    split,
    train_it=True,
    mini_factor=None,
    black_padding=False,
    center_idx=9,
    point_nb=600,
    sides="both",
    meta={},
    max_queries=[
        TransQueries.affinetrans,
        TransQueries.images,
        TransQueries.verts3d,
        TransQueries.center3d,
        TransQueries.joints3d,
        TransQueries.camintrs,
        BaseQueries.sides,
        BaseQueries.features
    ],
    use_cache=True,
    limit_size=None,
    verts = True
):
    if dat_name == "dexycb":
        pose_dataset = dexycb.DexYCB(
            subject='s0', split=split, verts=verts)
    else:
        raise ValueError("Unrecognized dataset name {}".format(dat_name))

    # Find maximal dataset-compatible queries
    queries = set(max_queries).intersection(set(pose_dataset.all_queries))
    if dat_name == "stereohands":
        max_rot = np.pi
        scale_jittering = 0.2
        center_jittering = 0.2
    else:
        max_rot = np.pi
        scale_jittering = 0.3
        center_jittering = 0.2

    if "override_scale" not in meta:
        meta["override_scale"] = False
    dataset = HandDataset(
        pose_dataset,
        black_padding=black_padding,
        block_rot=False,
        sides=sides,
        train=train_it,
        max_rot=max_rot,
        normalize_img=False,
        center_idx=center_idx,
        point_nb=point_nb,
        scale_jittering=scale_jittering,
        center_jittering=center_jittering,
        queries=queries,
        as_obj_only=meta["override_scale"]
    )
    if limit_size is not None:
        if len(dataset) < limit_size:
            warnings.warn(
                "limit size {} < dataset size {}, working with full dataset".format(
                    limit_size, len(dataset)
                )
            )
        else:
            warnings.warn(
                "Working wth subset of {} of size {}".format(dat_name, limit_size)
            )
            dataset = Subset(dataset, list(range(limit_size)))
    return dataset
