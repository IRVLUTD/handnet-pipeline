#Image detection model for hands
import ros_demo
from PIL import Image
import glob

def run_network(self):
    # image path for dataset
    image_path = '../Downloads/imageDataset/0104T143721/*.jpg'
    image_files = glob.glob(image_path)
    for image_file in image_files:
        self.im = cv2.imread(image_file)

    # original code
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

# import glob
# import cv2

## create a list of image paths
# image_paths = glob.glob('../Downloads/imageDataset/0104T143721/*.jpg')

## create an instance of the class containing the run_network function
# network_runner = NetworkRunner()

## loop through each image and pass it to the run_network function
# for image_path in image_paths:
#     # read the image
#     im = cv2.imread(image_path)

#    # pass the image to the run_network function
#     network_runner.run_network(im)

