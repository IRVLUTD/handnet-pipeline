# DexYCB Toolkit
# Copyright (C) 2021 NVIDIA Corporation
# Licensed under the GNU General Public License v3.0 [see LICENSE for details]

"""Example of visualizing object and hand pose of one image sample."""

import numpy as np
import pyrender
import trimesh
import torch
import cv2
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

from manopth.manolayer import ManoLayer

from dex_ycb_toolkit.factory import get_dataset


def create_scene(sample, obj_file, pose, betas, side, faces):
  """Creates the pyrender scene of an image sample.

  Args:
    sample: A dictionary holding an image sample.
    obj_file: A dictionary holding the paths to YCB OBJ files.

  Returns:
    A pyrender scene object.
  """
  # Create pyrender scene.
  scene = pyrender.Scene(bg_color=np.array([0.0, 0.0, 0.0, 0.0]),
                         ambient_light=np.array([1.0, 1.0, 1.0]))

  # Add camera.
  fx = sample['intrinsics']['fx']
  fy = sample['intrinsics']['fy']
  cx = sample['intrinsics']['ppx']
  cy = sample['intrinsics']['ppy']
  cam = pyrender.IntrinsicsCamera(fx, fy, cx, cy)
  scene.add(cam, pose=np.eye(4))

  #Load poses.
  label = np.load(sample['label_file'])
  pose_y = label['pose_y']
  pose_m = label['pose_m']

  # Load YCB meshes.
  mesh_y = []
  for i in sample['ycb_ids']:
    mesh = trimesh.load(obj_file[i])
    mesh = pyrender.Mesh.from_trimesh(mesh)
    mesh_y.append(mesh)

  # Add YCB meshes.
  for o in range(len(pose_y)):
    if np.all(pose_y[o] == 0.0):
      continue
    pose_ycb = np.vstack((pose_y[o], np.array([[0, 0, 0, 1]], dtype=np.float32)))
    pose_ycb[1] *= -1
    pose_ycb[2] *= -1
    node = scene.add(mesh_y[o], pose=pose_ycb)



  mano_layer = ManoLayer(ncomps=30,
                         #flat_hand_mean=False,
                         #center_idx=9,
                         side=side[0],
                         mano_root='/home/cgalab/handobj/manopth/mano/models',
                         use_pca=True).cuda()
  pose_d = torch.from_numpy(pose_m).cuda()
  mano_layer_1 = ManoLayer(ncomps=30,
                         flat_hand_mean=False,
                         #center_idx=9,
                         side=side[0],
                         mano_root='/home/cgalab/handobj/manopth/mano/models',
                         use_pca=True).cuda()
  verts1, _ = mano_layer(pose.cuda(), betas.cuda())
  verts2, joints2 = mano_layer_1(pose_d[:, 0:48], betas.cuda())
  # verts, _ = mano_layer(pose.cuda(), betas.cuda(), pose_d[:, 48:51])
  # Add MANO meshes.
  max_rot=np.pi
  rot = np.random.uniform(low=-max_rot, high=max_rot)
  rot_mat = np.array(
            [
                [np.cos(rot), -np.sin(rot), 0],
                [np.sin(rot), np.cos(rot), 0],
                [0, 0, 1],
            ]
        ).astype(np.float32)
  inverse = np.linalg.inv(rot_mat)

  verts1 = verts1.view(778, 3)
  verts2 = verts2.view(778, 3)
  verts1 = verts1.cpu().detach().numpy().squeeze().squeeze()
  verts2 = verts2.cpu().detach().numpy().squeeze().squeeze()
  joints2 = joints2.cpu().detach().numpy().squeeze().squeeze()

  # verts1 = verts1.transpose()
  # verts1 = inverse.dot(verts1)
  # verts1 = verts1.transpose(1,0)

  # with open('joints2.npy', 'wb') as f:
  #   np.save(f, joints2)
  # verts2 = rot_mat.dot(
  #               verts2.transpose(1, 0)
  #           ).transpose()
  
  
  mesh1 = trimesh.Trimesh(vertices=verts1, faces=faces)
  mesh2 = trimesh.Trimesh(vertices=verts2, faces=faces)
  mesh1.export("./mesh1.obj")
  mesh2.export("./mesh2.obj")
  
  verts = verts.cpu().detach().numpy().squeeze().squeeze()
  # verts /= 1000
  # trans = np.eye(4)
  # trans[:3,:3] = [
  #   [  0.6135935, -0.1620590, -0.7728130],
  #  [0.1620590, -0.9320324,  0.3241180],
  #  [-0.7728130, -0.3241180, -0.5456259 ]
  # ]
  # # trans[:3,3]=pose_d[:, 48:51].cpu().detach().numpy().squeeze()
  # # trans[1, 3] *= -1
  # # trans[2, 3] *= -1
  # verts = trimesh.transformations.transform_points(
  #           verts,
  #           matrix=trans)

  
  # verts[:, 0] *= -1
  trans_2 = np.eye(4)
  # trans_2[:3,:3]=R.from_quat([ 0.7661432, -0.3830716, 0.3830716, -0.345741 ]).as_matrix()
  trans_2[:3,:3]=R.from_quat([ 0, 0, 1, 0.0 ]).as_matrix()
  verts = trimesh.transformations.transform_points(
            verts,
            matrix=trans_2)

  # verts[:, 0] *= -1
  # verts[:, 2] *= -1
  # verts[:, 1] *= -1
  # verts[:, 2] *= -1
  verts = torch.from_numpy(verts).cuda() + pose_d[:, 48:51].unsqueeze(1)*1000
  verts = verts.cpu().detach().numpy().squeeze().squeeze()
  # verts[:, 2] += 150
  verts /= 1000
  
  # verts[:, 1] *= -1
  # verts[:, 2] *= -1
  mesh = trimesh.Trimesh(vertices=verts, faces=faces)
  
  #mesh.apply_transform(trans)
  mesh1 = pyrender.Mesh.from_trimesh(mesh)
  mesh1.primitives[0].material.baseColorFactor = [0.7, 0.7, 0.7, 1.0]
  mesh2 = pyrender.Mesh.from_trimesh(mesh, wireframe=True)
  mesh2.primitives[0].material.baseColorFactor = [0.0, 0.0, 0.0, 1.0]
  node1 = scene.add(mesh1)
  node2 = scene.add(mesh2)

  # node1 = pyrender.Node(mesh=mesh1, rotation = np.array([ 0.8790978, 0, 0.4395489, -0.1843469 ]))
  # node2 = pyrender.Node(mesh=mesh2, rotation = np.array([ 0.8790978, 0, 0.4395489, -0.1843469 ]))
  # scene.add_node(node1)
  # scene.add_node(node2)

  return scene

def crop(orig_img, handbbx, scale, diagnostics = False):
    SMALLEST_SIZE = 256
    
    #assign the edges of the bounding box based on hand_dets
    xmin = handbbx[0]
    ymin = handbbx[1]
    xmax = handbbx[2]
    ymax = handbbx[3]

    #define the side length based on width and height
    width = abs(xmin-xmax)
    height = abs(ymin-ymax)
    side = int(max(width,height)*scale)
    side = max(side, SMALLEST_SIZE)

    #define the center of the bounding box
    xcenter = (xmax+xmin)/2
    ycenter = (ymax+ymin)/2
    
    #define the edges of the cropped image
    left_edge = max(0, int(xcenter - side/2))
    right_edge = min(len(orig_img[0])-1, int(xcenter + side/2))
    upper_edge = max(0, int(ycenter - side/2))
    bottom_edge = min(len(orig_img)-1, int(ycenter + side/2))

    img = orig_img[upper_edge:bottom_edge,left_edge:right_edge].copy()
    img = cv2.resize(img,(SMALLEST_SIZE,SMALLEST_SIZE),interpolation = cv2.INTER_LINEAR)
    
    if diagnostics:
        print("original image size :" + str(len(orig_img))+"x"+str(len(orig_img[0])))
        print("final image size :" + str(len(img))+"x"+str(len(img[0])))
        print("xcenter:"+str(xcenter)+" ycenter:"+str(ycenter)+" side:"+str(side))
        print("img["+str(upper_edge)+":"+str(bottom_edge)+", "+str(left_edge)+":"+str(right_edge)+"]")    

    return img

def main(name, idx, result, faces,  det):
  dataset = get_dataset(name)

  sample = dataset[idx]

  scene_r = create_scene(sample, dataset.obj_file, result["pose"], result["shape"], result["side"], faces)
  scene_v = create_scene(sample, dataset.obj_file, result["pose"], result["shape"], result["side"], faces)

  im_real = cv2.imread(sample['color_file'])
  #im_real = crop(im_real, det, 1.2)

  print('Visualizing pose in camera view using pyrender renderer')

  r = pyrender.OffscreenRenderer(viewport_width=im_real.shape[1], viewport_height=im_real.shape[0])

  im_render, _ = r.render(scene_r)
  im_real = im_real[:, :, ::-1]

  im = 0.33 * im_real.astype(np.float32) + 0.67 * im_render.astype(np.float32)
  im = im.astype(np.uint8)

  print('Close the window to continue.')

  plt.imshow(im)
  plt.tight_layout()
  #plt.show()

  print('Visualizing pose using pyrender 3D viewer')
  pyrender.Viewer(scene_v)



def create_scene_before(sample, obj_file):
  """Creates the pyrender scene of an image sample.
  Args:
    sample: A dictionary holding an image sample.
    obj_file: A dictionary holding the paths to YCB OBJ files.
  Returns:
    A pyrender scene object.
  """
  # Create pyrender scene.
  scene = pyrender.Scene(bg_color=np.array([0.0, 0.0, 0.0, 0.0]),
                         ambient_light=np.array([1.0, 1.0, 1.0]))

  # Add camera.
  fx = sample['intrinsics']['fx']
  fy = sample['intrinsics']['fy']
  cx = sample['intrinsics']['ppx']
  cy = sample['intrinsics']['ppy']
  cam = pyrender.IntrinsicsCamera(fx, fy, cx, cy)
  scene.add(cam, pose=np.eye(4))

  # Load poses.
  label = np.load(sample['label_file'])
  pose_y = label['pose_y']
  pose_m = label['pose_m']

  # Load YCB meshes.
  mesh_y = []
  for i in sample['ycb_ids']:
    mesh = trimesh.load(obj_file[i])
    mesh = pyrender.Mesh.from_trimesh(mesh)
    mesh_y.append(mesh)

  # Add YCB meshes.
  for o in range(len(pose_y)):
    if np.all(pose_y[o] == 0.0):
      continue
    pose = np.vstack((pose_y[o], np.array([[0, 0, 0, 1]], dtype=np.float32)))
    pose[1] *= -1
    pose[2] *= -1
    node = scene.add(mesh_y[o], pose=pose)

  # Load MANO layer.
  mano_layer = ManoLayer(flat_hand_mean=False,
                         ncomps=30,
                         side=sample['mano_side'][0],
                         use_pca=True)
  faces = mano_layer.th_faces.numpy()
  betas = torch.tensor(sample['mano_betas'], dtype=torch.float32).unsqueeze(0)

  # Add MANO meshes.
  if not np.all(pose_m == 0.0):
    pose = torch.from_numpy(pose_m)
    vert, _ = mano_layer(pose[:, 0:48], betas, pose[:, 48:51])
    vert /= 1000
    vert = vert.view(778, 3)
    vert = vert.numpy()
    vert[:, 1] *= -1
    vert[:, 2] *= -1
    mesh = trimesh.Trimesh(vertices=vert, faces=faces)
    mesh1 = pyrender.Mesh.from_trimesh(mesh)
    mesh1.primitives[0].material.baseColorFactor = [0.7, 0.7, 0.7, 1.0]
    mesh2 = pyrender.Mesh.from_trimesh(mesh, wireframe=True)
    mesh2.primitives[0].material.baseColorFactor = [0.0, 0.0, 0.0, 1.0]
    node1 = pyrender.Node(mesh=mesh1, rotation = np.array([0.0, 0.0, 1.0, 1.0]))
    node2 = pyrender.Node(mesh=mesh2, rotation = np.array([0.0, 0.0, 1.0, 1.0]))
    scene.add_node(node1)
    scene.add_node(node2)

  return scene


def main_before():
  name = 's0_test'
  dataset = get_dataset(name)

  idx = 70

  sample = dataset[idx]

  scene_r = create_scene_before(sample, dataset.obj_file)
  scene_v = create_scene_before(sample, dataset.obj_file)

  print('Visualizing pose in camera view using pyrender renderer')

  r = pyrender.OffscreenRenderer(viewport_width=dataset.w,
                                 viewport_height=dataset.h)

  im_render, _ = r.render(scene_r)

  im_real = cv2.imread(sample['color_file'])
  im_real = im_real[:, :, ::-1]

  im = 0.33 * im_real.astype(np.float32) + 0.67 * im_render.astype(np.float32)
  im = im.astype(np.uint8)

  print('Close the window to continue.')

  plt.imshow(im)
  plt.tight_layout()
  plt.show()

  print('Visualizing pose using pyrender 3D viewer')

  pyrender.Viewer(scene_v)


if __name__ == '__main__':
  main_before()
