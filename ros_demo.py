

"""Test HN pipeline on ros images"""

from tomlkit import key
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
from handnet_pipeline.handnet_pipeline import HandNet
import message_filters
import cv2
import threading
import argparse
import numpy as np
import rospy

from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
from utils.vistool import VisualUtil
lock = threading.Lock()


import os
import torch
import torch.nn as nn
import torch.optim as optim
import colorsys

import models
from core.config import cfg
from aug_utils import j2d_processing
from coord_utils import get_bbox, process_bbox
from funcs_utils import load_checkpoint, save_obj
from graph_utils import build_coarse_graphs
from vis import vis_2d_keypoints
from _mano import MANO
from a2j.a2j import convert_joints

import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'# 'osmesa'
import torch
from torchvision.utils import make_grid
import numpy as np
import pyrender
import trimesh
import math

from pyrender.constants import RenderFlags

class Renderer:
    def __init__(self, faces, resolution=(224,224), orig_img=False, wireframe=False):
        self.resolution = resolution

        self.faces = faces
        self.orig_img = orig_img
        self.wireframe = wireframe
        self.renderer = pyrender.OffscreenRenderer(
            viewport_width=self.resolution[0],
            viewport_height=self.resolution[1],
        )

        # set the scene
        self.scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0], ambient_light=(0.3, 0.3, 0.3))

        # light = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=0.8)
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=0.8)

        light_pose = np.eye(4)
        light_pose[:3, 3] = [0, -1, 1]
        self.scene.add(light, pose=light_pose)

        light_pose[:3, 3] = [0, 1, 1]
        self.scene.add(light, pose=light_pose)

        light_pose[:3, 3] = [1, 1, 2]
        self.scene.add(light, pose=light_pose)

    def render(self, img, params, verts, mesh_filename=None, color=(1.0, 1.0, 0.9, 1.0)):

        mesh = trimesh.Trimesh(vertices=verts, faces=self.faces)

        if mesh_filename is not None:
            mesh.export(mesh_filename)

        camera = pyrender.IntrinsicsCamera(*params)

        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.0,
            alphaMode='OPAQUE',
            baseColorFactor=color
        )

        mesh = pyrender.Mesh.from_trimesh(mesh, material=material, smooth=False)

        mesh_node = self.scene.add(mesh, 'mesh')

        camera_pose = np.eye(4)
        cam_node = self.scene.add(camera, pose=camera_pose)

        if self.wireframe:
            render_flags = RenderFlags.RGBA | RenderFlags.ALL_WIREFRAME
        else:
            render_flags = RenderFlags.RGBA

        rgb, depth = self.renderer.render(self.scene, flags=render_flags)
        valid_mask = (depth > 0)[:, :, np.newaxis]
        output_img = rgb[:, :, :3] * valid_mask + (1 - valid_mask) * img
        image = output_img.astype(np.uint8)

        self.scene.remove_node(mesh_node)
        self.scene.remove_node(cam_node)

        return image

def render(result, params, orig_height, orig_width, orig_img, mesh_face, mesh_filename=None):
    pred_verts = result['mesh']

    # Setup renderer for visualization
    renderer = Renderer(mesh_face, resolution=(orig_width, orig_height), orig_img=True)
    rendered_img = renderer.render(
        orig_img,
        params,
        pred_verts,
        mesh_filename=mesh_filename,
    )

    return rendered_img


def get_joint_setting(mesh_model):
    joint_regressor = mesh_model.joint_regressor
    joint_num = 21
    skeleton = ( (0,1), (0,5), (0,9), (0,13), (0,17), (1,2), (2,3), (3,4), (5,6), (6,7), (7,8), (9,10), (10,11), (11,12), (13,14), (14,15), (15,16), (17,18), (18,19), (19,20) )
    hori_conn = (
    (1, 5), (5, 9), (9, 13), (13, 17), (2, 6), (6, 10), (10, 14), (14, 18), (3, 7), (7, 11), (11, 15), (15, 19),
    (4, 8), (8, 12), (12, 16), (16, 20))
    graph_Adj, graph_L, graph_perm, graph_perm_reverse = \
        build_coarse_graphs(mesh_model.face, joint_num, skeleton, hori_conn, levels=6)
    model_chk_path = './experiment/pose2mesh_manoJ_train_freihand/final.pth.tar'

    model = models.pose2mesh_net.get_model(joint_num, graph_L)
    checkpoint = load_checkpoint(load_dir=model_chk_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    return model, joint_regressor, joint_num, skeleton, graph_L, graph_perm_reverse

def predict_mesh(model, joint_input, graph_perm_reverse, mesh_model):
    bbox = get_bbox(joint_input)
    bbox2 = process_bbox(bbox.copy())
    joint_img, _ = j2d_processing(joint_input.copy(), (cfg.MODEL.input_shape[1], cfg.MODEL.input_shape[0]), bbox2, 0, 0, None)

    joint_img = joint_img[:, :2]
    joint_img /= np.array([[cfg.MODEL.input_shape[1], cfg.MODEL.input_shape[0]]])
    mean, std = np.mean(joint_img, axis=0), np.std(joint_img, axis=0)
    joint_img = (joint_img.copy() - mean) / std
    joint_img = torch.Tensor(joint_img[None, :, :]).cuda()

    # estimate mesh, pose
    model.eval()
    pred_mesh, _ = model(joint_img)
    pred_mesh = pred_mesh[:, graph_perm_reverse[:mesh_model.face.max() + 1], :]

    out = {}

    out['mesh'] = pred_mesh[0].detach().cpu().numpy()

    return out

class ImageListener:

    def __init__(self, network, RGBD=False, left=False):

        self.network = network
        self.cv_bridge = CvBridge()
        self.vistool = VisualUtil('dexycb')

        self.im = None
        self.depth = None
        self.rgb_frame_id = None
        self.rgb_frame_stamp = None
        self.empty_label = np.zeros((176, 176, 3), dtype=np.uint8)
        self.rgbd = RGBD
        self.left = left
        
        # initialize a node
        rospy.init_node("pose_rgb")
        self.box_pub = rospy.Publisher('box_label', Image, queue_size=10)
        self.label_pub = rospy.Publisher('pose_label', Image, queue_size=10)
        self.mesh_pub = rospy.Publisher('mesh_label', Image, queue_size=10)


        self.base_frame = 'measured/base_link'
        rgb_sub = message_filters.Subscriber('/head_camera/rgb/image_raw', Image, queue_size=10)
        depth_sub = message_filters.Subscriber('/head_camera/depth/image_raw', Image, queue_size=10)
        msg = rospy.wait_for_message('/head_camera/depth/camera_info', CameraInfo)
        self.camera_frame = 'measured/camera_color_optical_frame'
        self.target_frame = self.base_frame

        # update camera intrinsics
        intrinsics = np.array(msg.K).reshape(3, 3)
        self.paras = np.array([intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2] ,intrinsics[1, 2]])

        queue_size = 1
        slop_seconds = 0.1
        ts = message_filters.ApproximateTimeSynchronizer([rgb_sub, depth_sub], queue_size, slop_seconds)
        ts.registerCallback(self.callback_rgbd)

        # pose2mesh
        cfg.DATASET.target_joint_set = 'mano'
        cfg.MODEL.posenet_pretrained = False

        self.mesh_model = MANO()
        self.model, self.joint_regressor, self.joint_num, self.skeleton, self.graph_L, self.graph_perm_reverse = get_joint_setting(self.mesh_model)
        self.model = self.model.cuda()
        self.joint_regressor = torch.Tensor(self.joint_regressor).cuda()
        self.virtual_crop_size = 500

        # video callback
        # self.box_video = []
        # self.label_video = []
        # ts = message_filters.ApproximateTimeSynchronizer([self.box_pub, self.label_pub], queue_size, slop_seconds)
        # ts.registerCallback(self.callback_video)

    def callback_rgbd(self, rgb, depth):

        if depth.encoding == '32FC1':
            depth_cv = self.cv_bridge.imgmsg_to_cv2(depth)
        elif depth.encoding == '16UC1':
            depth_cv = self.cv_bridge.imgmsg_to_cv2(depth).copy().astype(np.float32)
            depth_cv /= 1000.0
        else:
            rospy.logerr_throttle(
                1, 'Unsupported depth type. Expected 16UC1 or 32FC1, got {}'.format(
                    depth.encoding))
            return

        im = self.cv_bridge.imgmsg_to_cv2(rgb, 'bgr8')

        with lock:
            self.im = im.copy()
            self.depth = depth_cv.copy()
            self.rgb_frame_id = rgb.header.frame_id
            self.rgb_frame_stamp = rgb.header.stamp


    def run_network(self):

        with lock:
            if listener.im is None:
              return
            if self.im is None: 
                return 
            im_color = self.im.copy()
            depth_img = self.depth.copy()
            rgb_frame_id = self.rgb_frame_id
            rgb_frame_stamp = self.rgb_frame_stamp

        # flip if using left hand
        if self.left:
            im_color = cv2.flip(im_color, 1)
            depth_img = cv2.flip(depth_img, 1)

        # run network
        with torch.inference_mode():
            im_color_forward = [torch.from_numpy(cv2.cvtColor(im_color, cv2.COLOR_BGR2RGB).transpose(2, 0, 1).astype(np.float32) / 255.0).cuda()]
            depth_img = torch.from_numpy(depth_img).unsqueeze(0).unsqueeze(0).cuda()
            if self.rgbd:
                im_rgbd = torch.cat([im_color_forward[0].unsqueeze(0), depth_img], dim=1)
            keypoint_pred, depth_im, detections = self.network(im_color_forward, depth_images=im_rgbd if self.rgbd else depth_img)
            keypoint_pred = keypoint_pred.cpu()
            depth_im = depth_im.cpu()
            detections = detections.cpu()

        # unbatch
        detection = detections[0].clone()
        keypoint_pred = keypoint_pred[0].clone()

        # clamping
        detection[:2] = torch.clamp(detection[:2], 0, im_color.shape[0])
        detection[2:] = torch.clamp(detection[2:], 0, im_color.shape[1])
        detection = detection.numpy()

        keypoint_pred = torch.clamp(keypoint_pred, min=0.0, max=176.0)
        keypoint_pred = keypoint_pred.cpu().numpy()

        # sanity checking

        joint_input = convert_joints(keypoint_pred, None, detection, None, 176, 176)[:, :2]
        bbox = get_bbox(joint_input)
        bbox2 = process_bbox(bbox.copy())
        print(keypoint_pred)

        if detections.max() == 0 or detections is None or bbox2 is None:
            label_msg = self.cv_bridge.cv2_to_imgmsg(self.empty_label)
            label_msg.header.stamp = rgb_frame_stamp
            label_msg.header.frame_id = rgb_frame_id
            label_msg.encoding = 'rgb8'
            self.label_pub.publish(label_msg)
            box_msg = self.cv_bridge.cv2_to_imgmsg(im_color)
            box_msg.header.stamp = rgb_frame_stamp
            box_msg.header.frame_id = rgb_frame_id
            box_msg.encoding = 'bgr8'
            self.box_pub.publish(box_msg)
            return

        # metrics
        depth_im = depth_im[0]

        image_to_draw = cv2.cvtColor(im_color, cv2.COLOR_BGR2RGB).copy()
        cv2.rectangle(image_to_draw, (detection[0], detection[1]), (detection[2], detection[3]), (0, 255, 0), 1)
        bbox_msg = self.cv_bridge.cv2_to_imgmsg(image_to_draw.astype(np.uint8))
        bbox_msg.header.stamp = rgb_frame_stamp
        bbox_msg.header.frame_id = rgb_frame_id
        bbox_msg.encoding = 'rgb8'
        self.box_pub.publish(bbox_msg)

        color_im_crop = cv2.cvtColor(cv2.resize(im_color[detection[1]:detection[3], detection[0]:detection[2], :], (176, 176)), cv2.COLOR_BGR2RGB).copy()

        # visualize and publish
        label = self.vistool.plot(color_im_crop, None, None, jt_uvd_pred=keypoint_pred, return_image=True)
        label_msg = self.cv_bridge.cv2_to_imgmsg(label.astype(np.uint8))
        label_msg.header.stamp = rgb_frame_stamp
        label_msg.header.frame_id = rgb_frame_id
        label_msg.encoding = 'rgb8'
        self.label_pub.publish(label_msg)

        full_image = cv2.cvtColor(im_color, cv2.COLOR_BGR2RGB).copy()
        joints2d = convert_joints(keypoint_pred, None, detection, None, 176, 176)[:, :2]
        joints3d = convert_joints(keypoint_pred, None, detection, self.paras, 176, 176)
        orig_height, orig_width = full_image.shape[:2]
        out = predict_mesh(self.model, joints2d, self.graph_perm_reverse, self.mesh_model)

        out['mesh'] = out['mesh'] * 1000. + joints3d[0]
        out['mesh'] /= 1000.
        out['mesh'][:, 1] *= -1
        out['mesh'][:, 2] *= -1

        if args.save_path:
            # find avail filename
            filename = 0
            while os.path.exists(os.path.join(self.save_path, f'{filename}.npy')):
                filename += 1
            np.save(os.path.join(self.save_path, f'{filename}.npy'), out['mesh'])

        # vis mesh (and optionally save)
        rendered_img = render(out, self.paras, orig_height, orig_width, full_image, self.mesh_model.face, mesh_filename=os.path.join(self.save_path, f'{filename}.obj' if args.save_path else None))
        rendered_img = self.cv_bridge.cv2_to_imgmsg(rendered_img.astype(np.uint8))
        rendered_img.header.stamp = rgb_frame_stamp
        rendered_img.header.frame_id = rgb_frame_id
        rendered_img.encoding = 'rgb8'
        self.mesh_pub.publish(rendered_img)

        # alternate vis 2d pose
        # tmpkps = np.zeros((3, len(joints2d)))
        # tmpkps[0, :], tmpkps[1, :], tmpkps[2, :] = joints2d[:, 0], joints2d[:, 1], 1
        # tmpimg = full_image.copy().astype(np.uint8)
        # pose_vis_img = vis_2d_keypoints(tmpimg, tmpkps, skeleton)
        # pose_vis_img = self.cv_bridge.cv2_to_imgmsg(pose_vis_img.astype(np.uint8))
        # pose_vis_img.header.stamp = rgb_frame_stamp
        # pose_vis_img.header.frame_id = rgb_frame_id
        # pose_vis_img.encoding = 'rgb8'
        # self.pose_pub.publish(pose_vis_img)

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Demo E2E-HandNet on ROS')
    parser.add_argument('--pretrained_fcos', dest='pretrained_fcos', help='Pretrained FCOS model',
                        default='models/fcos.pth', type=str)
    # parser.add_argument('--pretrained_a2j', dest='pretrained_a2j', help='Pretrained A2J model',
    #                     default='wandb/a2j/E2E-HandNet/326lfxim/checkpoints/epoch=44-step=128879.ckpt', type=str)
    parser.add_argument('--num_classes', dest='num_classes', help='Number of classes in FCOS model', default=3, type=int)
    parser.add_argument('--pretrained_a2j', dest='pretrained_a2j', help='Pretrained A2J model',
                    default='models/a2j.pth', type=str)
    parser.add_argument('--rgbd', dest='rgbd', help='Use RGBD', type=bool, default=False)
    parser.add_argument('--left', dest='left', help='Use left hand', type=bool, default=False)
    parser.add_argument('--save_path', dest='save_path', help='Path to save results', default='output/', type=str)

    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = parse_args()
    network = HandNet(args, reload_detector=True, num_classes=args.num_classes, reload_a2j=True, RGBD=args.rgbd).cuda().eval()
    cudnn.benchmark = True
    #network.eval()

    os.makedirs(args.save_path, exist_ok=True)

    # image listener
    listener = ImageListener(network, RGBD=args.rgbd, left=args.left)
    while not rospy.is_shutdown():
       listener.run_network()
    #listener.write_video('test_box.mp4', 'test_label.mp4')