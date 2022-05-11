

"""Test E2E-HN on ros images"""

import torch
import torch.nn.parallel
import torch.nn.functional as F
import torchvision.transforms as Transforms
import torch.backends.cudnn as cudnn
import torch.utils.data
from e2e_handnet.e2e_handnet import E2EHandNet
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


class ImageListener:

    def __init__(self, network, RGBD=False):

        self.network = network
        self.cv_bridge = CvBridge()
        self.vistool = VisualUtil('dexycb')

        self.im = None
        self.depth = None
        self.rgb_frame_id = None
        self.rgb_frame_stamp = None
        self.empty_label = np.zeros((176, 176, 3), dtype=np.uint8)
        self.rgbd = RGBD
        
        # initialize a node
        rospy.init_node("pose_rgb")
        self.box_pub = rospy.Publisher('box_label', Image, queue_size=10)
        self.label_pub = rospy.Publisher('pose_label', Image, queue_size=10)


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

        # video callback
        # self.box_video = []
        # self.label_video = []
        # ts = message_filters.ApproximateTimeSynchronizer([self.box_pub, self.label_pub], queue_size, slop_seconds)
        # ts.registerCallback(self.callback_video)

    def callback_video(self, box_msg, label_msg):
        box_frame = self.cv_bridge.imgmsg_to_cv2(box_msg, 'bgr8')
        label_frame = self.cv_bridge.imgmsg_to_cv2(label_msg, 'bgr8')

        with lock:
            self.box_video.append(box_frame)
            self.label_video.append(label_frame)

    def write_video(self, path_box, path_label):
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(path_box, fourcc, 30.0, (640, 480))
        for i in range(len(self.box_video)):
            out.write(self.box_video[i])
        out.release()

        out = cv2.VideoWriter(path_label, fourcc, 30.0, (176, 176))
        for i in range(len(self.label_video)):
            out.write(self.label_video[i])
        out.release()

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
            im_color = self.im.copy()
            depth_img = self.depth.copy()
            rgb_frame_id = self.rgb_frame_id
            rgb_frame_stamp = self.rgb_frame_stamp

        # run network
        with torch.inference_mode():
            im_color_forward = [torch.from_numpy(cv2.cvtColor(im_color, cv2.COLOR_BGR2RGB).transpose(2, 0, 1).astype(np.float32) / 255.0).cuda()]
            depth_img = torch.from_numpy(depth_img).unsqueeze(0).unsqueeze(0).cuda()
            if self.rgbd:
                im_rgbd = torch.cat([im_color_forward[0].unsqueeze(0), depth_img], dim=1)
            keypoint_pred, depth_im, detections = self.network(im_color_forward, depth_images=im_rgbd if self.rgbd else depth_img)

        if detections.max() == 0:
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

        # unbatch
        detection = detections[0]
        keypoint_pred = keypoint_pred[0]
        depth_im = depth_im[0]

        image_to_draw = Transforms.ToPILImage()(im_color_forward[0]).convert('RGB')
        image_to_draw = np.array(image_to_draw)
        cv2.rectangle(image_to_draw, (detection[0], detection[1]), (detection[2], detection[3]), (0, 255, 0), 1)
        bbox_msg = self.cv_bridge.cv2_to_imgmsg(image_to_draw.astype(np.uint8))
        bbox_msg.header.stamp = rgb_frame_stamp
        bbox_msg.header.frame_id = rgb_frame_id
        bbox_msg.encoding = 'rgb8'
        self.box_pub.publish(bbox_msg)

        color_im_crop = F.interpolate(im_color_forward[0][:,  detection[1]:detection[3] + 1, detection[0]:detection[2] + 1].unsqueeze(0), size=(176, 176)).squeeze(0)
        color_im_crop = Transforms.ToPILImage()(color_im_crop).convert('RGB')
        color_im_crop = np.array(color_im_crop)

        # visualize and publish
        label = self.vistool.plot(color_im_crop, None, None, jt_uvd_pred=keypoint_pred.cpu().numpy(), return_image=True)
        label_msg = self.cv_bridge.cv2_to_imgmsg(label.astype(np.uint8))
        label_msg.header.stamp = rgb_frame_stamp
        label_msg.header.frame_id = rgb_frame_id
        label_msg.encoding = 'rgb8'
        self.label_pub.publish(label_msg)


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Demo E2E-HandNet on ROS')
    parser.add_argument('--pretrained_fcos', dest='pretrained_fcos', help='Pretrained FCOS model',
                        default='models/fcos_handobj_100K_res34/detector_1_25.pth', type=str)
    parser.add_argument('--pretrained_a2j', dest='pretrained_a2j', help='Pretrained A2J model',
                        default='wandb/a2j/E2E-HandNet/326lfxim/checkpoints/epoch=44-step=128879.ckpt', type=str)

    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = parse_args()
    network = E2EHandNet(args, reload_detector=True, reload_a2j=True, RGBD=True).cuda().eval()
    cudnn.benchmark = True
    #network.eval()

    # image listener
    listener = ImageListener(network, RGBD=True)
    while not rospy.is_shutdown():
       listener.run_network()
    #listener.write_video('test_box.mp4', 'test_label.mp4')