import glob
from a2j.a2j import A2JModel
from tqdm import tqdm
import cv2
import numpy as np
import torch


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='A2J Inference on Depth Image Directory')

    parser.add_argument('--depth_image_dir', type=str, default='depth_images/',)
    parser.add_argument('--resume', type=str, default='models/a2j_dexycb_1/a2j_35.pth',)
    args = parser.parse_known_args()
    
    model = A2JModel(21, crop_height=176, crop_width=176)

    filenames = glob.glob(args.depth_image_dir + '/*.png')

    all_joints_uvd = np.zeros((len(filenames), 21, 3))

    with torch.inference_mode():
        for idx, depth_image_filename in enumerate(tqdm(filenames)):
            depth_image = cv2.imread(depth_image_filename, cv2.IMREAD_ANYDEPTH)
            depth_image = depth_image.astype(np.float32)
            depth_image = depth_image / 1000.0
            depth_image = torch.from_numpy(depth_image)
            depth_image = depth_image.unsqueeze(0)

            # forward pass through A2J to obtain keypoint predictions in uvd
            jt_uvd = model(depth_image)
            all_joints_uvd[idx] = jt_uvd.numpy()

    np.save('all_joints_uvd.npy', all_joints_uvd)
