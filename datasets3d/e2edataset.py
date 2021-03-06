import random
import traceback
import cv2

from dex_ycb_toolkit.factory import get_dataset
from torch.utils.data import Dataset
from PIL import Image
import torch
import numpy as np
import pickle
import os
from manopth.manolayer import ManoLayer
from tqdm import tqdm
from zmq import HANDSHAKE_IVL

from datasets3d.queries import TransQueries, BaseQueries
import pycocotools.mask

class E2EDataset(Dataset):
    """
    DexYCB EndtoEnd Dataset
    """

    def __init__(
        self,
        transform,
        train=True,
        val=False,
    ):
        # Dataset attributes
        if train:
            self.data = get_dataset('s0_train')
        elif val:
            self.data = get_dataset('s0_val')
        else:
            self.data = get_dataset('s0_test')

        self.transform = transform
        if train:
            self.refined_idx = pickle.load(open('data/e2e/cache/refined_train_idx.pkl', 'rb'))
        elif val:
            self.refined_idx = pickle.load(open('data/e2e/cache/refined_val_idx.pkl', 'rb'))
        else:
            self.refined_idx = pickle.load(open('data/e2e/cache/refined_test_idx.pkl', 'rb'))

        print("Loading 3D annotations")
        self.load_3d(train)

    def __len__(self):
        return len(self.refined_idx)

    def load_3d(self, train):
        split = 'train' if train else 'test'

        if os.path.exists(f'data/e2e/cache/{split}_3d_a2j.pt'):
            dict_3d = torch.load(f'data/e2e/cache/{split}_3d_a2j.pt')
            self.meshes = dict_3d['meshes']
            self.joints3d = dict_3d['joints3d']
            return


        self.mano_layer_left = ManoLayer(ncomps=45,
                        flat_hand_mean=False,
                        side='left',
                        mano_root='misc/mano/models',
                        use_pca=True)

        self.mano_layer_right = ManoLayer(ncomps=45,
                        flat_hand_mean=False,
                        side='right',
                        mano_root='misc/mano/models',
                        use_pca=True)
        
        self.joints3d = torch.zeros(len(self), 21, 3)
        self.meshes = torch.zeros((len(self), 778, 3))

        pose_left = torch.zeros((len(self), 48))
        pose_right = torch.zeros((len(self), 48))

        mask_left = torch.zeros(len(self), dtype=torch.bool)
        mask_right = torch.zeros(len(self), dtype=torch.bool)

        beta_left = torch.zeros((len(self), 10))
        beta_right = torch.zeros((len(self), 10))

        trans_left = torch.zeros((len(self), 3))
        trans_right = torch.zeros((len(self), 3))

        if os.path.exists(f'data/e2e/cache/{split}_labels_3d.pt'):
            labels_3d = torch.load(f'data/e2e/cache/{split}_labels_3d.pt')
            pose_left = labels_3d['pose']['left']
            pose_right = labels_3d['pose']['right']
            beta_left = labels_3d['beta']['left']
            beta_right = labels_3d['beta']['right']
            mask_left = labels_3d['mask']['left']
            mask_right = labels_3d['mask']['right']
            trans_left = labels_3d['trans']['left']
            trans_right = labels_3d['trans']['right']
        else:
            for idx, i in enumerate(tqdm((self.refined_idx))):
                sample = self.data[i]

                label = np.load(sample['label_file'])
                pose_m = label['pose_m']
                if not pose_m.any():
                    continue
                pose_d = torch.from_numpy(pose_m).squeeze()
                betas = torch.tensor(sample['mano_betas'], dtype=torch.float32)

                if sample['mano_side'] == 'left':
                    pose_left[idx] = pose_d[0:48]
                    beta_left[idx] = betas
                    mask_left[idx] = True
                    trans_left[idx] = pose_d[48:51]
                else:
                    pose_right[idx] = pose_d[0:48]
                    beta_right[idx] = betas
                    mask_right[idx] = True
                    trans_right[idx] = pose_d[48:51]

        # debug
        labels_3d = {
            'pose': {
                'left': pose_left,
                'right': pose_right
            },
            'beta':{
                'left': beta_left,
                'right': beta_right
            },
            'mask': {
                'left': mask_left,
                'right': mask_right
            },
            'trans':{
                'left': trans_left,
                'right': trans_right
            }
        }

        torch.save(labels_3d, f'data/e2e/cache/{split}_labels_3d.pt')

        verts3d_right, joints3d_right = self.mano_layer_right(pose_right[mask_right,:], beta_right[mask_right,:], trans_right[mask_right, :])
        verts3d_left, joints3d_left = self.mano_layer_left(pose_left[mask_left,:], beta_left[mask_left,:], trans_left[mask_left, :])

        self.joints3d[mask_right] = joints3d_right[:,:,:]
        self.meshes[mask_right] = verts3d_right[:,:,:]

        self.joints3d[mask_left] = joints3d_left[:,:,:]
        self.meshes[mask_left] = verts3d_left[:,:,:]

        dict_3d = {
            'meshes': self.meshes,
            'joints3d': self.joints3d
        }
        torch.save(dict_3d, f'data/e2e/cache/{split}_3d_a2j.pt')
        

    def get_sample(self, idx):
        source_sample = self.data[self.refined_idx[idx]]

        paras = np.array(list(source_sample['intrinsics'].values()))


        im = Image.open(source_sample['color_file'])
        depth = cv2.imread(source_sample['depth_file'], cv2.IMREAD_ANYDEPTH) / 1000.
        
        # 3D data
        verts3d = self.meshes[idx]
        joints3d = self.joints3d[idx]
        
        label = np.load(source_sample['label_file'])
        joints_2d = label['joint_2d']

        mask = label['seg'] == 255
        mask = np.asfortranarray(mask)
        rle = pycocotools.mask.encode(mask)
        bbox = np.array(pycocotools.mask.toBbox(rle).tolist())
        bbox[2:] += bbox[:2]

        source_idx = torch.tensor([idx], dtype=torch.int64)
        dexycb_id = torch.tensor([self.refined_idx[idx]], dtype=torch.int64)
        joints_2d = torch.from_numpy(joints_2d).squeeze()
        
        img_id = idx

        labels = []
        boxes = np.zeros((len(source_sample['ycb_ids'] + [255]), 4))

        label = np.load(source_sample['label_file'])
        for idx, y in enumerate(source_sample['ycb_ids'] + [255]):
            mask = label['seg'] == y
            if np.count_nonzero(mask) == 0:
                continue
            mask = np.asfortranarray(mask)
            rle = pycocotools.mask.encode(mask)
            bbox = np.array(pycocotools.mask.toBbox(rle).tolist())
            bbox[2:] += bbox[:2]
            if y == 255:
                category_id = 22
            else:
                category_id = y
            labels.append(category_id)
            boxes[idx] = bbox

        indexes = []
        for idx, i in enumerate(boxes):
            if i[0] == 0 and i[1] == 0 and i[2] == 0 and i[3] == 0:
                indexes.append(idx)

        boxes = np.delete(boxes, indexes, axis=0)
        handinfo = np.full((len(boxes), 5), -1., dtype=np.float32)
        handinfo[:, 4].fill(0)

        idx_hand = -1
        for idx in range(len(labels)):
            if labels[idx] == 22:
                idx_hand = idx
        if idx_hand >= 0:
            handinfo[idx_hand, 1] = 1 if source_sample['mano_side'] == 'right' else 0

        handbox = boxes[idx_hand]
        joints_2d -= handbox[:2]


        image_id = torch.tensor([img_id])
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        box_info = torch.as_tensor(handinfo, dtype=torch.float32)

        target = {}
        target["image_id"] = image_id
        target["boxes"] = boxes
        target["labels"] = labels
        target["box_info"] = box_info

        sample = {}
        sample["idx"] = source_idx
        sample["dexycb_id"] = dexycb_id
        sample[TransQueries.verts3d] = verts3d.float()
        sample[TransQueries.joints3d] = joints3d.float()
        sample[TransQueries.joints2d] = joints_2d.float()
        sample[BaseQueries.sides] = torch.tensor([0]) if source_sample['mano_side'] =='left' else torch.tensor([1])

        sample["box"] = torch.tensor(handbox, dtype=torch.float32)
        
        return self.transform(im), target, sample, depth[np.newaxis, :].astype(np.float32), paras.astype(np.float32)

    def get_height_and_width(self,idx):
        source_sample = self.data[self.refined_idx[idx]]
        im = Image.open(source_sample['color_file'])
        return im.height, im.width

    def __getitem__(self, idx):
        try:
            sample = self.get_sample(idx)
        except Exception:
            traceback.print_exc()
            print("Encountered error processing sample {}".format(idx))
            random_idx = random.randint(0, len(self))
            sample = self.get_sample(random_idx, self.queries)
        return sample