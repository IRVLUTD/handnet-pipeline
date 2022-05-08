import math
from ntpath import join
import random
import traceback

from dex_ycb_toolkit.factory import get_dataset
from manopth.manolayer import ManoLayer
from torch.utils.data import Dataset
from PIL import Image
import torch
import pycocotools.mask
import numpy as np
import pickle
import cv2
import os
from tqdm import tqdm
from utils.vistool import VisualUtil


def xyz2uvd(pts, paras, flipy=1):
    # paras: [fx, fy, fu, fv]
    pts_uvd = pts.copy()
    pts_uvd = pts_uvd.reshape(-1, 3)
    pts_uvd[:, 1] *= flipy
    pts_uvd[:, :2] = pts_uvd[:, :2] * paras[:2] / pts_uvd[:, 2:] + paras[2:]

    return pts_uvd.reshape(pts.shape).astype(np.float32)


def uvd2xyz(pts, paras, flipy=1):
    # paras: (fx, fy, fu, fv)
    pts_xyz = pts.copy()
    pts_xyz = pts_xyz.reshape(-1, 3)
    pts_xyz[:, :2] = (pts_xyz[:, :2] - paras[2:]) * pts_xyz[:, 2:] / paras[:2]
    pts_xyz[:, 1] *= flipy

    return pts_xyz.reshape(pts.shape).astype(np.float32)



class A2JDataset(Dataset):
    """
    DexYCB A2J Dataset
    """

    def __init__(
        self,
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
        if train:
            self.refined_idx = pickle.load(open('data/e2e/cache/refined_train_idx.pkl', 'rb'))
        else:
            self.refined_idx = pickle.load(open('data/e2e/cache/refined_test_idx.pkl', 'rb'))

        self.cropWidth = 176
        self.cropHeight = 176
        self.keypointsNumber = 21
        self.xy_thres = 110
        self.RandRotate = 180 
        self.RandScale = (1.0, 0.5)
        self.vistool = VisualUtil('dexycb')
        self.augment = train

        print("Loading 3D annotations")
        self.load_3d(train)

    def __len__(self):
        return len(self.refined_idx)

    def transform(self, img, label, matrix):
        '''
        img: [H, W]  label, [N,2]   
        '''
        img_out = cv2.warpAffine(img,matrix,(self.cropWidth,self.cropHeight))
        label_out = np.ones((self.keypointsNumber, 3))
        label_out[:,:2] = label[:,:2].copy()
        label_out = np.matmul(matrix, label_out.transpose())
        label_out = label_out.transpose()

        return img_out, label_out

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

    # add conditional augmentation for test
    def get_sample(self, idx):
        sample = self.data[self.refined_idx[idx]]
        
        im = cv2.imread(sample['depth_file'], cv2.IMREAD_ANYDEPTH) / 1000.
        color_im = cv2.imread(sample['color_file'])
        img_id = idx

        label = np.load(sample['label_file'])
        for idx, y in enumerate(sample['ycb_ids'] + [255]):
            mask = label['seg'] == y
            if np.count_nonzero(mask) == 0:
                continue
            mask = np.asfortranarray(mask)
            rle = pycocotools.mask.encode(mask)
            bbox = np.array(pycocotools.mask.toBbox(rle).tolist())
            bbox[2:] += bbox[:2]
            if y == 255:
                # padding
                w = bbox[2] - bbox[0]
                h = bbox[3] - bbox[1]
                percent = 0.3
                bbox[0] = max(0, bbox[0] - percent * (w))
                bbox[1] = max(0, bbox[1] - percent * (h))
                bbox[2] = min(im.shape[1], bbox[2] + percent * (w))
                bbox[3] = min(im.shape[0], bbox[3] + percent * (h))
                break

        dexycb_id = np.array([self.refined_idx[img_id]]).astype(np.int64)

        if self.augment:
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            RandCropShiftx = math.floor(0.1 * w)
            RandCropShifty = math.floor(0.1 * h)
            RandomOffset_1 = np.random.randint(-1*RandCropShiftx,RandCropShiftx) if RandCropShiftx > 0 else 0
            RandomOffset_2 = np.random.randint(-1*RandCropShifty,RandCropShifty) if RandCropShifty > 0 else 0
            RandomOffset_3 = np.random.randint(-1*RandCropShiftx,RandCropShiftx) if RandCropShiftx > 0 else 0
            RandomOffset_4 = np.random.randint(-1*RandCropShifty,RandCropShifty) if RandCropShifty > 0 else 0


            if (RandomOffset_1 > 0 and RandomOffset_3 < 0) or (RandomOffset_1 < 0 and RandomOffset_3 > 0):
                RandomOffset_1 =  0 - RandomOffset_1
            if (RandomOffset_2 > 0 and RandomOffset_4 < 0) or (RandomOffset_2 < 0 and RandomOffset_4 > 0):
                RandomOffset_2 =  0 - RandomOffset_2

            # maintain a shift in the same direction

            RandomRotate = np.random.randint(-1*self.RandRotate,self.RandRotate)
            matrix = cv2.getRotationMatrix2D((self.cropWidth/2,self.cropHeight/2),RandomRotate, 1.0)
        else:
            RandomOffset_1 = 0
            RandomOffset_2 = 0
            RandomOffset_3 = 0
            RandomOffset_4 = 0
            RandomRotate = 0
            matrix = cv2.getRotationMatrix2D((self.cropWidth/2,self.cropHeight/2),0, 1.0)

        new_Xmin = max(bbox[0] + RandomOffset_1, 0)
        new_Ymin = max(bbox[1] + RandomOffset_2, 0)
        new_Xmax = min(bbox[2] + RandomOffset_3, im.shape[1] - 1)
        new_Ymax = min(bbox[3] + RandomOffset_4, im.shape[0] - 1)

        imCrop = im[int(new_Ymin):int(new_Ymax), int(new_Xmin):int(new_Xmax)].copy()
        color_imCrop = color_im[int(new_Ymin):int(new_Ymax), int(new_Xmin):int(new_Xmax)].copy()
        try:
            imgResize = cv2.resize(imCrop, (self.cropWidth, self.cropHeight), interpolation=cv2.INTER_NEAREST)
            color_imgResize = cv2.resize(color_imCrop, (self.cropWidth, self.cropHeight), interpolation=cv2.INTER_NEAREST)
        except cv2.error as e:
            print("imCrop too small")

        imgResize = np.asarray(imgResize,dtype = 'float32')
        color_imgResize = np.asarray(color_imgResize,dtype = 'float32')

        joints_xyz = self.joints3d[img_id].numpy() / 1000.
        paras = np.array(list(sample['intrinsics'].values()))
        joints_uvd = np.ones_like(joints_xyz)
        joints_uvd[:, 0] = (xyz2uvd(joints_xyz, paras)[:, 0] - new_Xmin) * self.cropWidth / (new_Xmax - new_Xmin)
        joints_uvd[:, 1] = (xyz2uvd(joints_xyz, paras)[:, 1] - new_Ymin) * self.cropHeight / (new_Ymax - new_Ymin)
       
        if self.augment:
            imgResize, joints_uvd[:, :2] = self.transform(imgResize, joints_uvd, matrix)
            color_imgResize = cv2.warpAffine(color_imgResize,matrix,(self.cropWidth,self.cropHeight))
        joints_uvd[:, 2] = xyz2uvd(joints_xyz, paras)[:, 2]

        return imgResize[np.newaxis, :].astype(np.float32), joints_uvd, dexycb_id, color_imgResize.astype(np.float32), np.array([new_Xmin, new_Ymin, new_Xmax, new_Ymax]).astype(np.float32), paras.astype(np.float32)

    def __getitem__(self, idx):
        try:
            sample = self.get_sample(idx)
        except Exception:
            traceback.print_exc()
            print("Encountered error processing sample {}".format(idx))
            random_idx = random.randint(0, len(self))
            sample = self.get_sample(random_idx)
        return sample