from collections import OrderedDict, defaultdict
import json
from functools import lru_cache, wraps
import math
import os
import pickle
import math
import random

from matplotlib import pyplot as plt
import numpy as np
from PIL import Image, ImageFile
from scipy.spatial.distance import cdist
from scipy.io import loadmat
import torch
from torch.nn.modules.module import T
import trimesh
from tqdm import tqdm
import sys
from manopth.manolayer import ManoLayer

from datasets3d.queries import (BaseQueries, TransQueries,
                                        get_trans_queries)
from mano.webuser.smpl_handpca_wrapper_HAND_only import load_model as load_mano_model
from dex_ycb_toolkit.factory import get_dataset
import cv2
from mano_train.demo.preprocess import preprocess_frame

ImageFile.LOAD_TRUNCATED_IMAGES = True


class DexYCB():
    def __init__(self,
                 subject,
                 split,
                 verts=True):

        super().__init__()
        self.all_queries = [
            BaseQueries.images, BaseQueries.joints3d, BaseQueries.sides, TransQueries.joints2d
        ]
        # BaseQueries.depth,
        if verts:
            self.all_queries.append(BaseQueries.verts3d)
        trans_queries = get_trans_queries(self.all_queries)
        self.all_queries.extend(trans_queries)
        self.split = split
        self.subject = subject

        # Set cache path
        self.cache_folder = os.path.join('data', 'cache', 'dexycb')
        os.makedirs(self.cache_folder, exist_ok=True)
        self.det_folder = os.path.join('data', 'cache', 'dexycb', 'dets', self.subject)
        os.makedirs(self.det_folder, exist_ok=True)
        
        self.name = 'dexycb'

        self.load_dataset()

        self.mano_layer_left = ManoLayer(ncomps=45,
                        flat_hand_mean=False,
                        #center_idx*10=9,
                        side='left',
                        mano_root='/home/cgalab/handobj/manopth/mano/models',
                        use_pca=True)

        self.mano_layer_right = ManoLayer(ncomps=45,
                        flat_hand_mean=False,
                        #center_idx*10=9,
                        side='right',
                        mano_root='/home/cgalab/handobj/manopth/mano/models',
                        use_pca=True)

        self.parent_dir = '/home/cgalab/DataSets/dex-ycb/20200709-subject-01/augmented/'
        self.parent_dir = os.path.join(self.parent_dir, self.split)

        print('Loaded {} samples'.format(
            len(self.dataset)))

    def load_dataset(self):
        iternum = 1
        while os.path.exists(os.path.join(self.cache_folder, self.split + str(iternum) + '.pkl')):
            iternum+=1

        cache_path = os.path.join(self.cache_folder, self.split + str(iternum-1) + '.pkl')
        corrupt = False

        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as cache_f:
                try:
                    annotations = pickle.load(cache_f)
                except:
                    corrupt = True
            if not corrupt:
                print('Cached information for dataset {} loaded from {}'.format(
                self.name, cache_path))

        if corrupt or not os.path.exists(cache_path):
            # with open(os.path.join(self.root, 'paths.pkl'), 'rb') as p_f:
            #     paths = pickle.load(p_f)
            # imgs = np.load(os.path.join(self.root, 'core50_imgs.npz'))['x']
            
            hand_sides = []
            img_names = []
            joints = []
            seg = []
            
            dataset = get_dataset(f'{self.subject}_{self.split}')

            
            #img_names = self.detection(dataset)


            # for i in range(len(dataset)):
            #     print(i)
                # sample = dataset[i]
                # label = np.load(sample['label_file'])
                # pose_m = label['pose_m']
                # if not pose_m.any():
                #     continue
                # pose_d = torch.from_numpy(pose_m).cuda()
                # betas = torch.tensor(sample['mano_betas'], dtype=torch.float32).unsqueeze(0)
                # if sample['mano_side'] == 'left':
                #     _, joints3d = self.mano_layer_left(pose_d[:, 0:48], betas.cuda())
                # else:
                #     _, joints3d = self.mano_layer_right(pose_d[:, 0:48], betas.cuda())
            #     joints.append(label['joint_3d'])
            #     hand_sides.append(sample['mano_side'])
            
            annotations = {
                'dataset': dataset
            }

            with open(cache_path, 'wb') as fid:
                pickle.dump(annotations, fid)
            print('Wrote cache for dataset {} to {}'.format(self.name, cache_path))

        # Get image paths
        print(cache_path)
        # self.image_names = annotations['image_names']
        # self.hand_sides = annotations['hand_sides']
        # self.seg = annotations['seg']
        # self.joints = annotations['joints']
        self.dataset = annotations['dataset']

    def get_meta(self, idx):
        sample_dir = os.path.join(self.parent_dir, str(idx*10))
        with open(os.path.join(sample_dir, 'meta.pkl'), 'rb') as f:
            meta = pickle.load(f)
        return meta

    def get_data(self, idx):
        sample_dir = os.path.join(self.parent_dir, str(idx*10))
        data = np.load(os.path.join(sample_dir, 'data.npz'))

        image_path = os.path.join(sample_dir, 'img.png')
        img = Image.open(image_path)
        img = img.convert('RGB')
        return data, img


    def get_image(self, idx):
        image_path = self.dataset[idx*10]['color_file']
        img = Image.open(image_path)
        img = img.convert('RGB')

        # depth_image_path = self.dataset[idx*10]['depth_file']
        # depth_img = cv2.imread(depth_image_path, cv2.IMREAD_GRAYSCALE)
        crop_coord = self.crop_coord(idx*10)
        img = img.crop(crop_coord)
        # img = img.resize((256, 256))
        return crop_coord[0], crop_coord[1], img, 0

    def get_uncropped_image(self, idx):
        image_path = self.dataset[idx*10]['color_file']
        img = Image.open(image_path)
        img = img.convert('RGB')
        return img
    
    def get_backproj(self, idx):
        image_path = self.dataset[idx*10]['depth_file']
        img = cv2.imread(image_path, cv2.IMREAD_ANYDEPTH)
        K = self.dataset[idx*10]['intrinsics']
        K = np.array([[K['fx'], 0,  K['ppx'],], [0, K['fy'] , K['ppy'],], [0, 0, 1]])
        return img, K
    
    def get_joints2d(self,idx):
        sample = self.dataset[idx*10]
        label = np.load(sample['label_file'])
        joints2d = label['joint_2d'].squeeze()
        seg = label['seg']
        minus_one = joints2d + 1


        if not minus_one.any() or 255 not in seg:
            return minus_one
        else:
            return joints2d

    def get_joints3d(self, idx):
        sample = self.dataset[idx*10]
        label = np.load(sample['label_file'])
        return label['joint_3d'].squeeze()      

    def get_annotations(self, idx):
        sample = self.dataset[idx*10]
        label = np.load(sample['label_file'])
        joints3d = label['joint_3d'].squeeze()
        
        pose_m = label['pose_m']
        betas_m = torch.tensor(sample['mano_betas'], dtype=torch.float32).unsqueeze(0)
        pose_d = torch.from_numpy(pose_m)
        verts3d = np.zeros((778, 3))

        joints2d = label['joint_2d'].squeeze()
        seg = label['seg']

        # joints2d, trans, label_file, verts, pose, joints3d

        minus_one = joints2d + 1
        if not minus_one.any() or 255 not in seg:
            return (minus_one, '', verts3d, pose_m[:, :48], joints3d)
        else: 
            if sample['mano_side'] == 'left':
                verts3d, joints3d = self.mano_layer_left(pose_d[:, 0:48], betas_m)
            else:
                verts3d, joints3d = self.mano_layer_right(pose_d[:, 0:48], betas_m)
            verts3d = verts3d.view(778, 3).float().numpy()

            return (joints2d, sample['label_file'], verts3d, pose_m[:, :48], joints3d.cpu().detach().numpy().squeeze().squeeze())

    def get_sides(self, idx):
        side = self.dataset[idx*10]['mano_side']
        return side

    def __len__(self):
        return math.floor(len(self.dataset)/10)

    def crop_coord(self, idx):
        left = sys.maxsize
        right = 0
        top = sys.maxsize
        bottom = 0
        label = np.load(self.dataset[idx]['label_file'])
        seg = label['seg']
        for i in range(len(seg)):
            for j in range(len(seg[i])):
                if seg[i][j] == 255:
                    left = min(left, j)
                    right = max(right, j)
                    top = min(top, i)
                    bottom = max(bottom, i)
        if left == sys.maxsize or top == sys.maxsize:
            return (0, 0, 1, 1)      
        left = max(left - 20, 0)
        right = min(right + 20, seg.shape[1])
        top = max(top - 20, 0)
        bottom = min(bottom + 20, seg.shape[0])

        h = bottom - top
        w = right - left

        if w > h:
            diff = w - h
            top_sub = diff/2
            bottom_add = diff/2

            if bottom_add > seg.shape[0] - bottom:
                top_sub += bottom_add - (seg.shape[0] - bottom)
                bottom_add = seg.shape[0] - bottom
            if top_sub > top:
                bottom_add += top_sub - top
                top_sub = top

            top = top - top_sub
            bottom = bottom + bottom_add

        elif h > w:
            diff = h - w
            left_sub = diff/2
            right_add = diff/2

            if right_add > seg.shape[1] - right:
                left_sub += right_add - (seg.shape[1] - right)
                right_add = seg.shape[1] - right
            if left_sub > left:
                right_add += left_sub - left
                left_sub = left

            left = left - left_sub
            right = right + right_add
            
        #assert math.floor(h) == math.floor(w)

        return math.floor(left), math.floor(top), math.floor(right), math.floor(bottom)