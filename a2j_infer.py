import colorsys
import glob
from a2j.a2j import A2JModel
from tqdm import tqdm
import cv2
import numpy as np
import torch
from ros_demo import get_joint_setting, predict_mesh, render

from utils.utils import vis_minibatch
from utils.vistool import VisualUtil
from a2j.a2j import A2JModelLightning
from utils.utils import get_e2e_loaders


if __name__ == "__main__":
    vistool = VisualUtil('dexycb')
    import argparse
    parser = argparse.ArgumentParser(description='A2J Inference on Depth Image Directory')

    parser.add_argument('--depth_image_dir', type=str, default='/home/neilsong/Desktop/A2J_inputs/117222250549',)
    parser.add_argument('--resume', type=str, default='models/a2j.pth',)
    args = parser.parse_args()
    
    model = A2JModel(21, crop_height=176, crop_width=176, is_RGBD=False).cuda().eval()
    # model = A2JModelLightning.load_from_checkpoint(args.resume).cuda().eval()
    checkpoint = torch.load(args.resume, map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=False)

    filenames = glob.glob(args.depth_image_dir + '/crop_depth_right*.png')
    color_filenames = glob.glob(args.depth_image_dir  + '/crop_color_right*.jpg')

    filenames = sorted(filenames)
    color_filenames = sorted(color_filenames)

    all_joints_uvd = np.zeros((len(filenames), 21, 3))
    class a: pass
    a.workers = 1
    a.batch_size = 64
    a.aspect_ratio_group_factor = 0
    _, test, _, = get_e2e_loaders(a, a2j=True, distributed=False)
    
    for idx, (depth_image_filename, color_im_filename) in enumerate(tqdm(zip(filenames, color_filenames))):
        color_im = cv2.imread(color_im_filename)
        depth_image = cv2.imread(depth_image_filename, cv2.IMREAD_ANYDEPTH) / 1000. #be sure this is in millimeters
        color_im = cv2.resize(color_im, (176, 176))
        depth_image = cv2.resize(depth_image, (176, 176))
        depth_image = depth_image[np.newaxis, :].astype(np.float32)
        depth_image = depth_image
        depth_image = torch.from_numpy(depth_image).cuda()
        depth_image = depth_image.unsqueeze(0)

        color_forward = torch.from_numpy(color_im.transpose(2, 0, 1).astype(np.float32) / 255.0).unsqueeze(0).cuda()

        #combined_forward = torch.cat((color_forward, depth_image), dim=1)

        # forward pass through A2J to obtain keypoint predictions in uvd
        with torch.inference_mode():
            jt_uvd = model(depth_image)[0]
            all_joints_uvd[idx] = jt_uvd.cpu().numpy()

        vis_minibatch(
            color_im[np.newaxis, :],
            depth_image.detach().cpu().numpy(),
            all_joints_uvd[idx][np.newaxis, :],
            vistool,
            [1],
            path=f'/home/neilsong/handobj/e2e_handnet/out/test_{idx}.jpg',
        )


    np.save('all_joints_uvd.npy', all_joints_uvd)