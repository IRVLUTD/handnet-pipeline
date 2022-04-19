import argparse
import cv2

from dex_ycb_toolkit.coco_eval import COCOEvaluator
from datasets3d.a2jdataset import A2JDataset
import numpy as np


def main():
  data = A2JDataset(train=False)
  gt_joints_uvd_all = np.zeros((50, 21, 3))
  for i in range(50):
    image, gt_joints_uvd, _, color_im, _, _ = data[i]
    image *= 1000
    image = image.astype(np.uint16)
    cv2.imwrite(f'test/{i}.png', image[0])
    cv2.imwrite(f'test/color/{i}_color.png', color_im)
    gt_joints_uvd_all[i] = gt_joints_uvd
  np.save('gt_joints_uvd_all.npy', gt_joints_uvd_all)


if __name__ == '__main__':
  main()