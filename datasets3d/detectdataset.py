import random
import traceback

from dex_ycb_toolkit.factory import get_dataset
from torch.utils.data import Dataset
from PIL import Image
import torch
import pycocotools.mask
import numpy as np
import pickle

class DetectDataset(Dataset):
    """
    DexYCB Detection Dataset
    """

    def __init__(
        self,
        transform,
        train=True
    ):
        # Dataset attributes
        if train:
            self.data = get_dataset('s0_train')
        else:
            self.data = get_dataset('s0_test')

        self.transform = transform
        if train:
            self.refined_idx = pickle.load(open('data/e2e/cache/refined_train_idx.pkl', 'rb'))
        else:
            self.refined_idx = pickle.load(open('data/e2e/cache/refined_test_idx.pkl', 'rb'))

    def __len__(self):
        return len(self.refined_idx)

    def get_sample(self, idx):
        sample = self.data[self.refined_idx[idx]]
        
        im = Image.open(sample['color_file']).convert('RGB')
        img_id = idx

        labels = []
        boxes = np.zeros((1, 4))

        # only use hand bbox
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
                category_id = 0
            else:
                continue
            labels.append(category_id)
            boxes[0] = bbox

        indexes = []
        for idx, i in enumerate(boxes):
            if i[0] == 0 and i[1] == 0 and i[2] == 0 and i[3] == 0:
                indexes.append(idx)

        boxes = np.delete(boxes, indexes, axis=0)
        handinfo = np.full((len(boxes), 5), -1., dtype=np.float32)
        handinfo[:, 4].fill(0)

        idx_hand = -1
        for idx in range(len(labels)):
            if labels[idx] == 0:
                idx_hand = idx
        if idx_hand >= 0:
            handinfo[idx_hand, 1] = 1 if sample['mano_side'] == 'right' else 0


        image_id = torch.tensor([img_id])
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        box_info = torch.as_tensor(handinfo, dtype=torch.float32)

        target = {}
        target['dexycb_id'] = torch.tensor([self.refined_idx[img_id]], dtype=torch.int64)
        target["image_id"] = image_id
        target["boxes"] = boxes
        target["labels"] = labels
        target["box_info"] = box_info
        
        
        return self.transform(im), target

    def __getitem__(self, idx):
        try:
            sample = self.get_sample(idx)
        except Exception:
            traceback.print_exc()
            print("Encountered error processing sample {}".format(idx))
            random_idx = random.randint(0, len(self))
            sample = self.get_sample(random_idx)
        return sample