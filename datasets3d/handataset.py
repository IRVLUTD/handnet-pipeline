import random
import traceback

from matplotlib import pyplot as plt
import numpy as np
from PIL import Image, ImageFilter
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as func_transforms

from datasets3d import imgtrans, handutils, viz2d, vertexsample
from datasets3d.queries import (
    BaseQueries,
    TransQueries,
)


def bbox_from_joints(joints):
    x_min, y_min = joints.min(0)
    x_max, y_max = joints.max(0)
    bbox = [x_min, y_min, x_max, y_max]
    return bbox


class HandDataset(Dataset):
    """Class inherited by hands datasets
    hands datasets must implement the following methods:
    - get_image
    that respectively return a PIL image and a numpy array
    - the __len__ method

    and expose the following attributes:
    - the cache_folder : the path to the cache folder of the dataset
    """

    def __init__(
        self,
        pose_dataset,
        center_idx=9,
        point_nb=600,
        inp_res=256,
        max_rot=np.pi,
        normalize_img=False,
        split="train",
        scale_jittering=0.3,
        center_jittering=0.2,
        train=True,
        hue=0.15,
        saturation=0.5,
        contrast=0.5,
        brightness=0.5,
        blur_radius=0.5,
        queries=[
            BaseQueries.images,
            TransQueries.joints2d,
            TransQueries.verts3d,
            TransQueries.joints3d,
        ],
        sides="both",
        block_rot=False,
        black_padding=False,
        as_obj_only=False,
    ):
        """
        Args:
        center_idx: idx of joint on which to center 3d pose
        as_obj_only: apply same centering and scaling as when objects are
            not present
        sides: if both, don't flip hands, if 'right' flip all left hands to
            right hands, if 'left', do the opposite
        """
        # Dataset attributes
        self.pose_dataset = pose_dataset
        self.as_obj_only = as_obj_only
        self.inp_res = inp_res
        self.point_nb = point_nb
        self.normalize_img = normalize_img
        self.center_idx = center_idx
        self.sides = sides
        self.black_padding = black_padding

        # Color jitter attributes
        self.hue = hue
        self.contrast = contrast
        self.brightness = brightness
        self.saturation = saturation
        self.blur_radius = blur_radius

        self.max_rot = max_rot
        self.block_rot = block_rot

        # Training attributes
        self.train = train
        self.scale_jittering = scale_jittering
        self.center_jittering = center_jittering

        self.queries = queries

    def __len__(self):
        return len(self.pose_dataset)

    def get_sample(self, idx, query=None):
        sample = {}

        if idx == 74:
            print("")

        meta = self.pose_dataset.get_meta(idx)

        sample[BaseQueries.sides] = meta["hand_side"]
        sample["idx"] = meta["idx"]
        sample["label_file"] = meta["label_file"]

        data, img = self.pose_dataset.get_data(idx)
        joints2d = data["joints2d"]
        joints3d = data["joints3d"]
        verts3d = data["verts3d"]


        img_trans = func_transforms.to_tensor(img).float()
        img_trans = func_transforms.normalize(
            img_trans, [0.5, 0.5, 0.5], [1, 1, 1]
        )
        if TransQueries.images in query:
            sample[TransQueries.images] = img_trans
            # sample[BaseQueries.images] = img

        if TransQueries.joints2d in query:
            sample[TransQueries.joints2d] = torch.from_numpy(joints2d).float()

    
        if TransQueries.joints3d in query:
            sample[TransQueries.joints3d] = torch.from_numpy(
                joints3d
            )
    
        if TransQueries.verts3d in query:
            sample[TransQueries.verts3d] = torch.from_numpy(verts3d).float()

        return sample

    def __getitem__(self, idx):
        try:
            sample = self.get_sample(idx, self.queries)
        except Exception:
            traceback.print_exc()
            print("Encountered error processing sample {}".format(idx))
            random_idx = random.randint(0, len(self))
            sample = self.get_sample(random_idx, self.queries)
        return sample

    def visualize_original(self, idx, joint_idxs=False):
        queries = [
            BaseQueries.sides,
            BaseQueries.images,
            BaseQueries.joints2d,
            BaseQueries.objpoints2d,
            BaseQueries.camintrs,
            BaseQueries.objverts3d,
            BaseQueries.objfaces,
        ]
        sample_queries = []
        for query in queries:
            if query in self.pose_dataset.all_queries:
                sample_queries.append(query)
        sample = self.get_sample(idx, query=sample_queries)
        img = sample[BaseQueries.images]
        fig = plt.figure()
        ax = fig.add_subplot(111)
        # Add title
        if BaseQueries.sides in sample:
            side = sample[BaseQueries.sides]
            ax.set_title("{} hand".format(side))
        ax.imshow(img)
        if BaseQueries.joints2d in sample:
            joints = sample[BaseQueries.joints2d]
            # Scatter hand joints on image
            viz2d.visualize_joints_2d(
                ax, joints, joint_idxs=False, links=self.pose_dataset.links
            )
            ax.axis("off")
        if BaseQueries.objpoints2d in sample:
            objpoints = sample[BaseQueries.objpoints2d]
            # Scatter hand joints on image
            ax.scatter(objpoints[:, 0], objpoints[:, 1], alpha=0.01)
        plt.show()

    def display_proj(self, ax, sample, proj="z", joint_idxs=False):

        if proj == "z":
            proj_1 = 0
            proj_2 = 1
            ax.invert_yaxis()
        elif proj == "y":
            proj_1 = 0
            proj_2 = 2
        elif proj == "x":
            proj_1 = 1
            proj_2 = 2

        if TransQueries.joints3d in sample:
            joints3d = sample[TransQueries.joints3d]
            viz2d.visualize_joints_2d(
                ax,
                np.stack([joints3d[:, proj_1], joints3d[:, proj_2]], axis=1),
                joint_idxs=joint_idxs,
                links=self.pose_dataset.links,
            )
        # Scatter  projection of 3d vertices
        if TransQueries.verts3d in sample:
            verts3d = sample[TransQueries.verts3d]
            ax.scatter(verts3d[:, proj_1], verts3d[:, proj_2], s=1)

        ax.set_aspect("equal")  # Same axis orientation as imshow

    def visualize_3d_proj(self, idx, joint_idxs=False):
        queries = [
            BaseQueries.sides,
            BaseQueries.images,
            TransQueries.joints3d,
            TransQueries.images,
            TransQueries.verts3d,
            BaseQueries.joints2d,
        ]

        sample_queries = []
        for query in queries:
            if query in self.pose_dataset.all_queries:
                sample_queries.append(query)

        sample = self.get_sample(idx, query=sample_queries)
        fig = plt.figure()

        # Display transformed image
        ax = fig.add_subplot(121)
        img = sample[TransQueries.images].numpy().transpose(1, 2, 0)
        if not self.normalize_img:
            img += 0.5
        ax.imshow(img)

        # Add title
        if BaseQueries.sides in sample:
            side = sample[BaseQueries.sides]
            ax.set_title("{} hand".format(side))
        plt.show()

    def visualize_3d_transformed(self, idx, joint_idxs=False):
        queries = [
            BaseQueries.images,
            BaseQueries.sides,
            TransQueries.joints3d,
            TransQueries.images,
            TransQueries.verts3d,
            BaseQueries.joints2d,
        ]

        sample_queries = []
        for query in queries:
            if query in self.pose_dataset.all_queries:
                sample_queries.append(query)

        sample = self.get_sample(idx, query=sample_queries)
        fig = plt.figure()

        # Display transformed image
        ax = fig.add_subplot(141)
        img = sample[TransQueries.images].numpy().transpose(1, 2, 0)
        if not self.normalize_img:
            img += 0.5
        ax.imshow(img)
        # Add title
        if BaseQueries.sides in sample:
            side = sample[BaseQueries.sides]
            ax.set_title("{} hand".format(side))

        # Display XY projection
        ax = fig.add_subplot(142)
        self.display_proj(ax, sample, proj="z", joint_idxs=joint_idxs)

        # Display YZ projection
        ax = fig.add_subplot(143)
        self.display_proj(ax, sample, proj="x", joint_idxs=joint_idxs)

        # Display XZ projection
        ax = fig.add_subplot(144)
        self.display_proj(ax, sample, proj="y", joint_idxs=joint_idxs)
        plt.show()
        return fig

    def visualize_transformed(self, idx, joint_idxs=False):
        queries = [
            BaseQueries.images,
            BaseQueries.joints2d,
            BaseQueries.sides,
            TransQueries.images,
            TransQueries.joints2d,
            TransQueries.camintrs,
            TransQueries.center3d,
        ]
        sample_queries = []
        for query in queries:
            if query in self.pose_dataset.all_queries:
                sample_queries.append(query)
        sample = self.get_sample(idx, query=sample_queries)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        img = sample[TransQueries.images].numpy().transpose(1, 2, 0)
        if not self.normalize_img:
            img += 0.5
        ax.imshow(img)
        # Add title
        if BaseQueries.sides in sample:
            side = sample[BaseQueries.sides]
            ax.set_title("{} hand".format(side))

        if TransQueries.joints2d in sample:
            joints2d = sample[TransQueries.joints2d]
            viz2d.visualize_joints_2d(
                ax,
                joints2d,
                joint_idxs=joint_idxs,
                links=self.pose_dataset.links,
            )
        plt.show()


def cam_equal_aspect_3d(ax, verts, flip_x=False):
    """
    Centers view on cuboid containing hand and flips y and z axis
    and fixes azimuth
    """
    extents = np.stack([verts.min(0), verts.max(0)], axis=1)
    sz = extents[:, 1] - extents[:, 0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize / 2
    if flip_x:
        ax.set_xlim(centers[0] + r, centers[0] - r)
    else:
        ax.set_xlim(centers[0] - r, centers[0] + r)
    # Invert y and z axis
    ax.set_ylim(centers[1] + r, centers[1] - r)
    ax.set_zlim(centers[2] + r, centers[2] - r)