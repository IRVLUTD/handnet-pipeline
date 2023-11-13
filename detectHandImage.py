"""Test HandNet pipeline on images"""

import torch
import argparse
import cv2
import numpy as np
from handnet_pipeline.handnet_pipeline import HandNet
from pose2mesh.lib.funcs_utils import load_checkpoint
from pose2mesh.lib.graph_utils import build_coarse_graphs
from pose2mesh.lib.coord_utils import get_bbox, process_bbox
from pose2mesh.lib.vis import vis_2d_keypoints
from pose2mesh.lib._mano import MANO
from a2j.a2j import convert_joints
from pose2mesh.lib.core.config import cfg
from pose2mesh.lib.aug_utils import j2d_processing
from pose2mesh.lib.models.pose2mesh_net import get_model
import models

def predict_mesh(model, joint_input, graph_perm_reverse, mesh_model):
    bbox = get_bbox(joint_input)
    bbox2 = process_bbox(bbox.copy())
    joint_img, _ = j2d_processing(joint_input.copy(), (cfg.MODEL.input_shape[1], cfg.MODEL.input_shape[0]), bbox2, 0, 0,
                                  None)

    joint_img = joint_img[:, :2]
    joint_img /= np.array([[cfg.MODEL.input_shape[1], cfg.MODEL.input_shape[0]]])
    mean, std = np.mean(joint_img, axis=0), np.std(joint_img, axis=0)
    joint_img = (joint_img.copy() - mean) / std
    joint_img = torch.Tensor(joint_img[None, :, :]).cuda()
    model.eval()
    pred_mesh, _ = model(joint_img)
    pred_mesh = pred_mesh[:, graph_perm_reverse[:mesh_model.face.max() + 1], :]
    out = {}
    out['mesh'] = pred_mesh[0].detach().cpu().numpy()
    return out

def get_joint_setting(mesh_model):
    joint_regressor = mesh_model.joint_regressor
    joint_num = 21
    skeleton = (
    (0, 1), (0, 5), (0, 9), (0, 13), (0, 17), (1, 2), (2, 3), (3, 4), (5, 6), (6, 7), (7, 8), (9, 10), (10, 11),
    (11, 12), (13, 14), (14, 15), (15, 16), (17, 18), (18, 19), (19, 20))
    hori_conn = (
        (1, 5), (5, 9), (9, 13), (13, 17), (2, 6), (6, 10), (10, 14), (14, 18), (3, 7), (7, 11), (11, 15), (15, 19),
        (4, 8), (8, 12), (12, 16), (16, 20))
    graph_Adj, graph_L, graph_perm, graph_perm_reverse = \
        build_coarse_graphs(mesh_model.face, joint_num, skeleton, hori_conn, levels=6)
    model_chk_path = './experiment/pose2mesh_manoJ_train_freihand/final.pth.tar'

    model = get_model(joint_num, graph_L)
    checkpoint = load_checkpoint(load_dir=model_chk_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    return model, joint_regressor, joint_num, skeleton, graph_L, graph_perm_reverse

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Demo E2E-HandNet on ROS')
    parser.add_argument('--pretrained_fcos', dest='pretrained_fcos', help='Pretrained FCOS model',
                        default='models/fcos.pth', type=str)
    # parser.add_argument('--pretrained_a2j', dest='pretrained_a2j', help='Pretrained A2J model',
    #                     default='wandb/a2j/E2E-HandNet/326lfxim/checkpoints/epoch=44-step=128879.ckpt', type=str)
    parser.add_argument('--num_classes', dest='num_classes', help='Number of classes in FCOS model', default=3,
                        type=int)
    parser.add_argument('--pretrained_a2j', dest='pretrained_a2j', help='Pretrained A2J model',
                        default='models/a2j.pth', type=str)
    parser.add_argument('--rgbd', dest='rgbd', help='Use RGBD', type=bool, default=False)
    parser.add_argument('--left', dest='left', help='Use left hand', type=bool, default=False)
    parser.add_argument('--save_path', dest='save_path', help='Path to save results', default='output/', type=str)

    args = parser.parse_args()
    return args

args = parse_args()

class ImageProcessor:
    def __init__(self, network):
        self.network = network
        self.mesh_model = MANO()
        self.model, self.joint_regressor, self.joint_num, self.skeleton, self.graph_L, self.graph_perm_reverse = get_joint_setting(
            self.mesh_model)
        self.model = self.model.cuda()
        self.joint_regressor = torch.Tensor(self.joint_regressor).cuda()

    def process_image(self, image_path):
        im_color = cv2.imread(image_path)
        # run network
        with torch.inference_mode():
            im_color_forward = [torch.from_numpy(
                cv2.cvtColor(im_color, cv2.COLOR_BGR2RGB).transpose(2, 0, 1).astype(np.float32) / 255.0).cpu()]
            keypoint_pred, _, detections = self.network(im_color_forward)
            keypoint_pred = keypoint_pred.cpu().numpy()[0]
        # [post-processing from the original run_network function, excluding real-time related components]
        detection = detections[0].clone().numpy()
        keypoint_pred = np.clip(keypoint_pred, 0.0, 176.0)

        joints2d = convert_joints(keypoint_pred, None, detection, None, 176, 176)[:, :2]
        orig_height, orig_width = im_color.shape[:2]
        out = predict_mesh(self.model, joints2d, self.graph_perm_reverse, self.mesh_model)
        return out['mesh']
        processed_image_path = './assets/test_1.jpg'
        cv2.imwrite(processed_image_path, im_color)
        return out['mesh'], processed_image_path

def main():
    network = HandNet(args, reload_detector=True, num_classes=args.num_classes, reload_a2j=True, RGBD=args.rgbd).cpu().eval()
    processor = ImageProcessor(network)
    image_path = "./assets/test_1.jpg"
    mesh = processor.process_image(image_path)
    processed_image_path = processor.process_image(image_path)
    print(f"Processed mesh: {mesh}")
    #print(f"Image saved at: {processed_image_path}")

if __name__ == '__main__':
    main()
