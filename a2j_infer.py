import glob
from a2j.a2j import A2JModel
from tqdm import tqdm
import cv2
import numpy as np
import torch

from utils.utils import vis_minibatch
from utils.vistool import VisualUtil


if __name__ == "__main__":
    vistool = VisualUtil('dexycb')
    import argparse
    parser = argparse.ArgumentParser(description='A2J Inference on Depth Image Directory')

    parser.add_argument('--depth_image_dir', type=str, default='test/',)
    parser.add_argument('--resume', type=str, default='models/a2j_dexycb_1/a2j_35.pth',)
    args = parser.parse_args()
    
    model = A2JModel(21, crop_height=176, crop_width=176).cuda().eval()
    checkpoint = torch.load(args.resume, map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=False)

    filenames = glob.glob(args.depth_image_dir + '/*.png')

    all_joints_uvd = np.zeros((len(filenames), 21, 3))

    with torch.inference_mode():
        for idx, (depth_image_filename) in enumerate(tqdm(filenames)):
            depth_image = cv2.imread(depth_image_filename, cv2.IMREAD_ANYDEPTH) / 1000. #be sure this is in millimeters (uint16 precision)
            depth_image = depth_image[np.newaxis, :].astype(np.float32)
            depth_image = depth_image
            depth_image = torch.from_numpy(depth_image).cuda()
            depth_image = depth_image.unsqueeze(0)

            # forward pass through A2J to obtain keypoint predictions in uvd
            jt_uvd = model(depth_image)[0]
            all_joints_uvd[idx] = jt_uvd.cpu().numpy()

            # if (idx == 0):
            #     vis_minibatch(
            #         color_im[np.newaxis, :],
            #         depth_image.detach().cpu().numpy(),
            #         jt_uvd_gt[idx],
            #         vistool,
            #         [1],
            #         path='test.jpg',
            #         jt_pred=all_joints_uvd[idx][np.newaxis, :]
            #     )

    np.save('all_joints_uvd.npy', all_joints_uvd)