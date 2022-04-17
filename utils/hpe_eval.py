#from hpe_eval import *

from dex_ycb_toolkit.hpe_eval import HPEEvaluator
import os, pickle
from mano_train.exputils.monitoring import Monitor
import matplotlib.pyplot as plt
from mano_train.visualize.displaymano import *


def main(max_epoch, eval):
  #args = parse_args()
  for epoch in range(1, max_epoch):
    eval.load_model(epoch)
    eval.e_test_metrics(epoch, joints2d=True)

def hpe_plot(model_dir, start, end, level=None):
  out_dir = os.path.join(model_dir,'dexycb_metrics/', level) if level else os.path.join(model_dir,'dexycb_metrics/')

  with open(os.path.join(out_dir, 'hpe_epoch_metrics.pkl'), 'rb') as f:
    epoch_metrics = pickle.load(f)


  hosting_folder = os.path.join(out_dir, 'hosting/')
  os.makedirs(out_dir, exist_ok=True)
  monitor = Monitor(out_dir, hosting_folder=hosting_folder)

  save_dict = {
    'absolute': {},
    'root-relative' : {},
    'procrustes' : {},
  }

  for i in range(start, end + 1):
    save_dict['absolute']['test'] = epoch_metrics['ab'][f'{i}'][1]
    save_dict['root-relative']['test'] = epoch_metrics['rr'][f'{i}'][1]
    save_dict['procrustes']['test'] = epoch_metrics['pa'][f'{i}'][1]

    monitor.metrics.save_metrics(i, save_dict)
  monitor.metrics.plot_metrics(end)
  

def hpe_evaluate(model_name, model_dir, start, end, level=None):
  out_dir = os.path.join(model_dir,'dexycb_metrics/', level) if level is not None else os.path.join(model_dir,'dexycb_metrics/')
  hpe_eval = HPEEvaluator('s0_test')

  os.makedirs(out_dir, exist_ok=True)

  for epoch in range(start, end + 1):
    hpe_eval.evaluate(epoch, os.path.join(model_dir, f'{model_name}_test_metrics/level_{level}/s0_test_{epoch}.txt'), out_dir=out_dir) if level is not None else hpe_eval.evaluate(epoch, os.path.join(model_dir, f'{model_name}_test_metrics/s0_test_{epoch}.txt'), out_dir=out_dir)
  hpe_eval.save_epoch_metrics(out_dir)
  #plot(max_epoch)


state = ['Absolute', 'Root-relative', 'Procrustes']

def project(vis_dir, id, images, gt_joints3d_arr, pred_joints3d_arr, row_factor = 1, col_nb = 4, batch_nb = 3):
  fig = plt.figure(figsize=(12, 12))
  for row_idx in range(batch_nb):
    input_img = images
    # Show input image
    ax = fig.add_subplot(
        batch_nb * row_factor, col_nb, row_idx * row_factor * col_nb + 1
    )
    ax.imshow(input_img)
    ax.set_title(f"{str(id)} - {state[row_idx]}")
    ax.axis("off")
    # Show x, y projection
    ax = fig.add_subplot(
        batch_nb * row_factor, col_nb, row_idx * row_factor * col_nb + 2
    )
    add_joints_proj(ax, gt_joints3d_arr[row_idx], pred_joints3d_arr[row_idx], proj="z")
    ax.invert_yaxis()

    # Show x, z projection
    ax = fig.add_subplot(
        batch_nb * row_factor, col_nb, row_idx * row_factor * col_nb + 3
    )
    add_joints_proj(ax, gt_joints3d_arr[row_idx], pred_joints3d_arr[row_idx], proj="y")

    # Show y, z projection
    ax = fig.add_subplot(
        batch_nb * row_factor, col_nb, row_idx * row_factor * col_nb + 4
    )
    add_joints_proj(ax, gt_joints3d_arr[row_idx], pred_joints3d_arr[row_idx], proj="x")
  
  plt.show()
  #plt.savefig(os.path.join(vis_dir, f'error_{str(id)}.jpg'), dpi=100)


# def visualize_error(model_dir, epoch, test=False):
#   plt.switch_backend("Qt5Agg")
#   if test:
#     main(epoch)

#   out_dir = os.path.join(model_dir,'dexycb_metrics/')

#   hpe_eval = HPEEvaluator('s0_test')
#   vis_dir = os.path.join(out_dir, f'error_vis_{epoch}/')
#   os.makedirs(out_dir, exist_ok=True)
#   os.makedirs(vis_dir, exist_ok=True)

#   joint_3d_gt = hpe_eval.return_joints(f'output/dexycb/checkpoint_dexycb_hourglass/joints3d/s0_test_{epoch}.txt', 
#                       out_dir=out_dir)

#   res = hpe_eval._load_results(f'output/dexycb/checkpoint_dexycb_hourglass/joints3d/s0_test_{epoch}.txt', root3d=False)

#   dexYCB = DexYCB('./data/dexycb', 'test', img_size=128, cube=[200, 200, 200])
#   samples = [50, 100, 250, 320, 400]
#   tru_idx = []
#   images = []
#   for i in np.random.permutation(len(dexYCB)):
#     sample = dexYCB[i]
#     im = sample[6]
#     tru_idx = int(sample[8])

#     im = np.clip(im, 0, 255)
#     im = im.astype(np.uint8)
#     # convert im to BGR
#     im = im[:, :, (2, 1, 0)]
#     images.append(im)

#     kpt_gt = joint_3d_gt[tru_idx]
#     kpt_pred = res[tru_idx]
#     kpt_gt_arr = [kpt_gt, kpt_gt - kpt_gt[0], kpt_gt]
#     kpt_pred_arr = [kpt_pred, kpt_pred - kpt_pred[0], hpe_eval.align_with_scale(kpt_gt, kpt_pred)]
#     project(vis_dir, tru_idx, im, kpt_gt_arr, kpt_pred_arr)