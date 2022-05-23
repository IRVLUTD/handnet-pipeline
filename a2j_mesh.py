
from core.config import cfg
from vis import vis_2d_keypoints
from _mano import MANO
from a2j.a2j import convert_joints, A2JModel
from tqdm import tqdm
import cv2
import numpy as np
import torch
from ros_demo import get_joint_setting, predict_mesh, render
import torchvision.transforms as T

from utils.utils import vis_minibatch
from utils.vistool import VisualUtil
from a2j.a2j import A2JModelLightning
from utils.utils import get_e2e_loaders
import colorsys
import os

if __name__ == "__main__":

    vistool = VisualUtil('dexycb')
    import argparse
    parser = argparse.ArgumentParser(description='Mesh Inference on Subset of DexYCB')

    parser.add_argument('--resume', type=str, default='models/a2j.pth',)
    args = parser.parse_args()
    
    a2j = A2JModel(21, crop_height=176, crop_width=176, is_RGBD=False).cuda().eval()
    # model = A2JModelLightning.load_from_checkpoint(args.resume).cuda().eval()
    checkpoint = torch.load(args.resume, map_location="cpu")
    a2j.load_state_dict(checkpoint["model"], strict=False)

    os.makedirs('out', exist_ok=True)

    class a: pass
    a.workers = 1
    a.batch_size = 64
    a.aspect_ratio_group_factor = 0
    _, test, _, = get_e2e_loaders(a, a2j=True, distributed=False)


    cfg.DATASET.target_joint_set = 'mano'
    cfg.MODEL.posenet_pretrained = False

    mesh_model = MANO()
    model, joint_regressor, joint_num, skeleton, graph_L, graph_perm_reverse = get_joint_setting(mesh_model)
    model = model.cuda()
    joint_regressor = torch.Tensor(joint_regressor).cuda()

    for idx, (depth_crop, jt_uvd_gt, dexycb_id, color_crop, box, paras, combine_crop) in enumerate(tqdm(test)):

        full_image = cv2.imread(test.dataset.data[test.dataset.refined_idx[idx]]['color_file'])
        box = box.reshape(4).numpy()
        paras = paras.reshape(4).numpy()

        with torch.inference_mode():
            jt_uvd = a2j(depth_crop.cuda())[0].cpu().numpy()
        # run pose2mesh
        # todo - use detections, parameters, full_image, color_crop, and depth_crop from dataloader instead of local cache

        vis_minibatch(
            np.array([ np.array(T.ToPILImage()(i)) for i in color_crop ]),
            depth_crop[0].cpu().numpy(),
            jt_uvd[np.newaxis, :],
            vistool,
            [1],
            path=f'/home/neilsong/handobj/new/e2e-handnet/out/a2j_vis_{idx}.jpg',
        )

        keypoint_pred = np.clip(jt_uvd, a_min=0.0, a_max=176.0)
        joints2d = convert_joints(keypoint_pred, None, box, None, 176, 176)[:, :2]
        joints3d = convert_joints(keypoint_pred, None, box, paras, 176, 176)
        orig_height, orig_width = full_image.shape[:2]
        out = predict_mesh(model, joints2d, graph_perm_reverse, mesh_model)

        out['mesh'] = out['mesh'] * 1000. + joints3d[0]
        out['mesh'] /= 1000.
        out['mesh'][:, 1] *= -1
        out['mesh'][:, 2] *= -1

        # vis mesh
        rendered_img = render(out, paras, orig_height, orig_width, full_image, mesh_model.face)
        cv2.imwrite(f'out/demo_mesh_{idx}.png', rendered_img)

        # vis 2d pose
        tmpkps = np.zeros((3, len(joints2d)))
        tmpkps[0, :], tmpkps[1, :], tmpkps[2, :] = joints2d[:, 0], joints2d[:, 1], 1
        tmpimg = full_image.copy().astype(np.uint8)
        pose_vis_img = vis_2d_keypoints(tmpimg, tmpkps, skeleton)
        cv2.imwrite(f'out/demo_pose_{idx}.png', pose_vis_img)
        
        if idx == 10:
            break