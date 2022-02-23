import pickle
import os


import torch

from mano_train.networks.handnet import HandNet
from mano_train.datautils import ConcatDataloader
from mano_train.netscripts.get_datasets import get_dataset
from mano_train.modelutils import modelio

from datasets3d.queries import BaseQueries, TransQueries


def save_obj(filename, verticies, faces):
    with open(filename, "w") as fp:
        for v in verticies:
            fp.write("v %f %f %f\n" % (v[0], v[1], v[2]))
        for f in faces + 1:  # Faces are 1-based, not 0-based in obj files
            fp.write("f %d %d %d\n" % (f[0], f[1], f[2]))


def get_opts(resume_checkpoint):
    if resume_checkpoint.endswith("tar"):
        resume_checkpoint = os.path.join(
            "/", *resume_checkpoint.split("/")[:-1]
        )
    opt_path = os.path.join(resume_checkpoint, "opt.pkl")
    with open(opt_path, "rb") as p_f:
        opts = pickle.load(p_f)
    return opts


def reload_model(
    model_path,
    checkpoint_opts,
    mano_root="misc/mano/models",
    ico_divisions=3,
    no_beta=False,
    no_joints=False,
    no_trans=False,
):
    if "absolute_lambda" not in checkpoint_opts:
        checkpoint_opts["absolute_lambda"] = 0
    if "mano_lambda_joints3d" not in checkpoint_opts:
        checkpoint_opts["mano_lambda_joints3d"] = False
    if "mano_lambda_joints2d" not in checkpoint_opts:
        checkpoint_opts["mano_lambda_joints2d"] = False
    if "mano_adapt_skeleton" not in checkpoint_opts:
        checkpoint_opts["mano_adapt_skeleton"] = False
    if "mano_use_pca" not in checkpoint_opts:
        checkpoint_opts["mano_use_pca"] = True
    mano_use_shape = checkpoint_opts["mano_use_shape"]
    mano_use_trans = checkpoint_opts["mano_use_trans"]

    if no_beta:
        mano_lambda_shape=None
    else:
        mano_lambda_shape = checkpoint_opts["mano_lambda_shape"]
    
    if no_joints:
        mano_lambda_joints3d=None
    else:
        mano_lambda_joints3d = checkpoint_opts["mano_lambda_joints3d"]

    if no_trans:
        mano_lambda_trans=None
    else:
        mano_lambda_trans = checkpoint_opts["mano_lambda_trans"]

    model = HandNet(
        absolute_lambda=checkpoint_opts["absolute_lambda"],
        mano_adapt_skeleton=checkpoint_opts["mano_adapt_skeleton"],
        mano_root=mano_root,
        mano_center_idx=checkpoint_opts["center_idx"],
        mano_comps=checkpoint_opts["mano_comps"],
        mano_neurons=checkpoint_opts["hidden_neurons"],
        mano_use_shape=mano_use_shape,
        mano_use_pca=checkpoint_opts["mano_use_pca"],
        mano_lambda_verts=checkpoint_opts["mano_lambda_verts"],
        mano_lambda_joints3d=mano_lambda_joints3d,
        mano_lambda_joints2d=checkpoint_opts["mano_lambda_joints2d"],
        mano_lambda_shape=mano_lambda_shape,
        mano_use_trans= mano_use_trans,
        mano_lambda_trans = mano_lambda_trans
    )
    model = torch.nn.DataParallel(model)
    modelio.load_checkpoint(model, resume_path=model_path, strict=False)
    return model

def get_loader(
    dataset_names,
    metas,
    checkpoint_opts,
    max_queries=[
        TransQueries.affinetrans,
        TransQueries.images,  # TransQueries.segms,
        TransQueries.verts3d,
        TransQueries.center3d,
        TransQueries.joints3d,
        TransQueries.camintrs,
        BaseQueries.sides,
        BaseQueries.features
    ],
    shuffle=False,
    mini_factor=0.01,
):
    loaders = []
    for dat, meta in zip(dataset_names, metas):
        dataset = get_dataset(
            dat,
            split=meta["split"],
            max_queries=max_queries,
            mini_factor=mini_factor,
            meta=meta,
            center_idx=checkpoint_opts["center_idx"],
            train_it=False,
            point_nb=642,
            sides="left",
        )
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=1, shuffle=shuffle, num_workers=0
        )
        loaders.append(loader)
    concat_loader = ConcatDataloader(loaders)
    return concat_loader
