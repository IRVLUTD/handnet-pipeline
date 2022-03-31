from matplotlib import pyplot as plt
from matplotlib.colors import rgb2hex
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np


from datasets3d.queries import TransQueries, BaseQueries
from datasets3d.viz3d import visualize_joints_3d
from datasets3d.viz2d import visualize_joints_2d



#used in epochpass3d.py
def visualize_batch(
    save_img_path,
    sample,
    results,
    fig=None,
    faces_right=None,
    faces_left=None,
    max_rows=4,
    joint_idxs=False,
):
    images = None
    if BaseQueries.features in sample:
        features = sample[BaseQueries.features]
        batch_nb = min(features.shape[0], max_rows)


    # Get hand verts
    # if TransQueries.verts3d in sample:
    #     gt_batchverts3d = sample[TransQueries.verts3d]
    # else:
    #     gt_batchverts3d = None
    refine = False
    if "verts" in results:
        pred_batchverts3d = results["verts"].cpu().detach().numpy()
    else:
        pred_batchverts3d = None
    if "verts_refine" in results:
        pred_batchverts3d_refine = results["verts"].cpu().detach().numpy()
        refine = True
    else:
        pred_batchverts3d_refine = None

    # Get hand joints
    if TransQueries.joints3d in sample:
        gt_batchjoints3d = sample[TransQueries.joints3d].cpu().numpy()
    else:
        gt_batchjoints3d = None
    if "joints" in results:
        pred_batchjoints3d = results["joints"].cpu().detach().numpy()
    else:
        pred_batchjoints3d = None
    if "joints_refine" in results:
        refine = True
        pred_batchjoints3d_refine = results["joints_refine"].cpu().detach().numpy()
    else:
        pred_batchjoints3d_refine = None

    if refine:
        batch_nb = int(batch_nb / 2)


    # Initialize figure
    if refine:
        row_factor = 2
    else:
        row_factor = 1
    if fig is None:
        fig = plt.figure(figsize=(12, 12))
    fig.clf()
    col_nb = 5
    if TransQueries.depth in sample and "render_depth" in results:
        col_nb = 6
        has_segm = True
        depth_target = sample[TransQueries.depth]
        depth_pred = results["render_depth"]
    else:
        has_segm = False

    if BaseQueries.sides in sample:
        sides = sample[BaseQueries.sides]
        idxs = sample["idx"]
    else:
        sides = None
        idx = None
    if TransQueries.joints2d in sample:
        gt_batchjoints2d = sample[TransQueries.joints2d].cpu().numpy()
    else:
        gt_batchjoints2d = None
    if "joints2d" in results:
        pred_batchjoints2d = results["joints2d"].detach().cpu().numpy()
    else:
        pred_batchjoints2d = None
    # Create figure
    if "center3d" in results:
        # Add absolute gt offset to predictions
        center3d_preds = results["center3d"].detach().cpu().numpy()
        if pred_batchjoints3d is not None:
            pred_batchjoints3d = pred_batchjoints3d + center3d_preds[:, np.newaxis]
        if pred_batchverts3d is not None:
            pred_batchverts3d = pred_batchverts3d + center3d_preds[:, np.newaxis]

    if TransQueries.center3d in sample and "center3d" in results:
        # Add absolute gt offset to ground truth
        center3d_gt = sample[TransQueries.center3d].numpy()


        if gt_batchjoints3d is not None:
            gt_batchjoints3d = gt_batchjoints3d + center3d_gt[:, np.newaxis]
    for row_idx in range(batch_nb):
        # Show input image
        if sides is not None:
            side = sides[row_idx]
        else:
            side = None
        if images is not None:
            input_img = images[row_idx].permute(1, 2, 0).numpy() + 0.5
            ax = fig.add_subplot(
                batch_nb * row_factor, col_nb, row_idx * row_factor * col_nb + 1
            )
            ax.imshow(input_img)
            if gt_batchjoints2d is not None:
                gt_joints2d = gt_batchjoints2d[row_idx]
                visualize_joints_2d(ax, gt_joints2d, joint_idxs=False, alpha=0.5)
            if pred_batchjoints2d is not None:
                pred_joints2d = pred_batchjoints2d[row_idx]
                visualize_joints_2d(ax, pred_joints2d, joint_idxs=False)
            if side is not None:
                if side is not None:
                    side_text = 'Left' if str(side.numpy()) == '0' else 'Right'
                    ax.set_title(f"{str(idxs[row_idx].numpy())} - {side_text}")
            ax.axis("off")

            # Show input image refine
            if refine:
                ax = fig.add_subplot(
                    batch_nb * row_factor,
                    col_nb,
                    (row_idx * row_factor + 1) * col_nb + 1,
                )
                ax.imshow(input_img)
                if side is not None:
                    side_text = 'Left' if str(side.numpy()) == '0' else 'Right'
                    ax.set_title(f"{str(idxs[row_idx].numpy())} - side_text")
            ax.axis("off")

        # Get sample infos
        gt_joints3d = get_row(gt_batchjoints3d, row_idx)
        pred_joints3d = get_row(pred_batchjoints3d, row_idx)

        verts3d = get_row(pred_batchverts3d, row_idx)

        if refine:
            pred_joints3d_refine = get_row(pred_batchjoints3d_refine, row_idx)

            verts3d_refine = get_row(pred_batchverts3d_refine, row_idx)
        # Show output mesh
        # ax = fig.add_subplot(
        #     batch_nb * row_factor,
        #     col_nb,
        #     row_idx * row_factor * col_nb + 2,
        #     projection="3d",
        # )

        # Show x, y projection
        ax = fig.add_subplot(
            batch_nb * row_factor, col_nb, row_idx * row_factor * col_nb + 3
        )
        add_joints_proj(ax, gt_joints3d, pred_joints3d, proj="z")

        ax.invert_yaxis()

        if refine:
            ax = fig.add_subplot(
                batch_nb * row_factor, col_nb, (row_idx * row_factor + 1) * col_nb + 3
            )
            add_joints_proj(ax, gt_joints3d, pred_joints3d_refine, proj="z")

            ax.invert_yaxis()

        # Show x, z projection
        ax = fig.add_subplot(
            batch_nb * row_factor, col_nb, row_idx * row_factor * col_nb + 4
        )
        add_joints_proj(ax, gt_joints3d, pred_joints3d, proj="y")


        if refine:
            ax = fig.add_subplot(
                batch_nb * row_factor, col_nb, (row_idx * row_factor + 1) * col_nb + 4
            )
            add_joints_proj(ax, gt_joints3d, pred_joints3d_refine, proj="y")


        # Show y, z projection
        ax = fig.add_subplot(
            batch_nb * row_factor, col_nb, row_idx * row_factor * col_nb + 5
        )
        add_joints_proj(ax, gt_joints3d, pred_joints3d, proj="x")


        if refine:
            ax = fig.add_subplot(
                batch_nb * row_factor, col_nb, (row_idx * row_factor + 1) * col_nb + 5
            )
            add_joints_proj(ax, gt_joints3d, pred_joints3d_refine, proj="x")

        if has_segm:
            ax = fig.add_subplot(
                batch_nb * row_factor, col_nb, row_idx * row_factor * col_nb + 6
            )
            ax.imshow(depth_pred[row_idx].detach(), alpha=1)
            ax.imshow(depth_target[row_idx], alpha=0.5)
    plt.savefig(save_img_path, dpi=100)

#used here
def get_row(batch_sample, idx):
    if batch_sample is not None:
        row_sample = batch_sample[idx]
    else:
        row_sample = None
    return row_sample

#used here
def get_proj_axis(proj="z"):
    if proj == "z":
        proj_1 = 0
        proj_2 = 1
    elif proj == "y":
        proj_1 = 0
        proj_2 = 2
    elif proj == "x":
        proj_1 = 1
        proj_2 = 2
    return proj_1, proj_2

#used here
def add_joints_proj(ax, gt_keypoints, pred_keypoints, proj="z", joint_idxs=False):
    proj_1, proj_2 = get_proj_axis(proj=proj)
    if gt_keypoints is not None:
        visualize_joints_2d(
            ax,
            np.stack([gt_keypoints[:, proj_1], gt_keypoints[:, proj_2]], axis=1),
            alpha=0.2,
            joint_idxs=joint_idxs,
        )
    if pred_keypoints is not None:
        visualize_joints_2d(
            ax,
            np.stack([pred_keypoints[:, proj_1], pred_keypoints[:, proj_2]], axis=1),
            joint_idxs=joint_idxs,
        )
    ax.set_aspect("equal")

#*******used in workers.py and here*******
def add_mesh(ax, verts, faces, flip_x=False, c="b", alpha=1):
    ax.view_init(elev=90, azim=-90)
    mesh = Poly3DCollection(verts[faces], alpha=alpha)
    if c == "b":
        face_color = (141 / 255, 184 / 255, 226 / 255)
        edge_color = (0 / 255, 0 / 255, 112 / 255)
    elif c == "r":
        face_color = (226 / 255, 141 / 255, 141 / 255)
        edge_color = (112 / 255, 0 / 255, 0 / 255)
    elif c == "viridis":
        face_color = plt.cm.viridis(np.linspace(0, 1, faces.shape[0]))
        edge_color = None
        edge_color = (0 / 255, 0 / 255, 112 / 255)
    elif c == "plasma":
        face_color = plt.cm.plasma(np.linspace(0, 1, faces.shape[0]))
        edge_color = None
        # edge_color = (0 / 255, 0 / 255, 112 / 255)
    else:
        face_color = c
        edge_color = c

    mesh.set_edgecolor(edge_color)
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)
    cam_equal_aspect_3d(ax, verts, flip_x=flip_x)
    plt.tight_layout()

#used in handataset.py, and here
def cam_equal_aspect_3d(ax, verts, flip_x=False):
    """
    Centers view on cuboid containing hand and flips y and z axis
    and fixes azimuth
    """
    extents = np.stack([verts.min(0), verts.max(0)], axis=1)
    sz = extents[:, 1] - extents[:, 0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize / 2
    # if flip_x:
    #     ax.set_xlim(centers[0] - r, centers[0] + r)
    #     ax.set_ylim(centers[1] - r, centers[1] + r)
    # else:
    #     ax.set_xlim(centers[0] + r, centers[0] - r)
    #     ax.set_ylim(centers[1] + r, centers[1] - r)
    ax.set_xlim(centers[0] - r, centers[0] + r)
    ax.set_ylim(centers[1] + r, centers[1] - r)
    ax.set_zlim(centers[2] + r, centers[2] - r)

#used in epochpass3d.py
def save_pck_img(thresholds, pck_values, auc_all, save_pck_file, overlay=None):
    """
    Args:
        auc_all (float): Area under the curve
    """
    plt.clf()
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.weight"] = "light"
    fontsize = 36
    markersize = 16
    plt.plot(thresholds, pck_values, "ro-", markersize=markersize, label="Ours")

    # Relative 3D for 12 sequences of stereohands
    if overlay == "stereo_all":
        plt.title("Stereo dataset (12 seq.)", fontsize=40)
        gan_thresh = [
            20.2020,
            22.2222,
            24.2424,
            26.2626,
            28.2828,
            30.3030,
            32.3232,
            34.3434,
            36.3636,
            38.3838,
            40.4040,
            42.4242,
            44.4444,
            46.4646,
            48.4848,
            50.5051,
        ]

        gan_accuracies = [
            0.4416,
            0.4772,
            0.5101,
            0.5410,
            0.5699,
            0.5968,
            0.6212,
            0.6445,
            0.6660,
            0.6858,
            0.7049,
            0.7229,
            0.7394,
            0.7550,
            0.7697,
            0.7835,
        ]
        plt.plot(
            gan_thresh, gan_accuracies, "bv-", markersize=markersize, label="Ganerated"
        )

    # Relative 3D for 2 sequences of stereohands
    elif overlay == "stereo_test":
        plt.title("Stereo dataset (2 seq.)", fontsize=40)
        gan_thresh = [
            19.1919,
            22.2222,
            25.2525,
            28.2828,
            31.3131,
            34.3434,
            37.3737,
            40.4040,
            43.4343,
            46.4646,
            49.4949,
        ]
        gan_accuracies_wo = [
            0.7031,
            0.7323,
            0.7586,
            0.7831,
            0.8056,
            0.8249,
            0.8424,
            0.8586,
            0.8728,
            0.8859,
            0.8972,
        ]
        gan_accuracies_w = [
            0.8713,
            0.9035,
            0.9271,
            0.9446,
            0.9574,
            0.9670,
            0.9741,
            0.9795,
            0.9833,
            0.9867,
            0.9895,
        ]
        plt.plot(
            gan_thresh,
            gan_accuracies_wo,
            "bv-",
            markersize=markersize,
            label="Ganerated wo",
        )
        plt.plot(
            gan_thresh,
            gan_accuracies_w,
            "c^-",
            markersize=markersize,
            label="Ganerated w",
        )
        zimmerman_thresh = [
            21.0526315789474,
            23.6842105263158,
            26.3157894736842,
            28.9473684210526,
            31.5789473684211,
            34.2105263157895,
            36.8421052631579,
            39.4736842105263,
            42.1052631578947,
            44.7368421052632,
            47.3684210526316,
            50,
        ]
        zimmerman_accs = [
            0.869888888888889,
            0.896873015873016,
            0.916849206349206,
            0.932142857142857,
            0.943507936507937,
            0.952753968253968,
            0.959904761904762,
            0.966047619047619,
            0.971595238095238,
            0.976547619047619,
            0.980174603174603,
            0.983277777777778,
        ]
        plt.plot(
            zimmerman_thresh, zimmerman_accs, "gs-", markersize=markersize, label="Z&B"
        )
        chpr_thresh = [20, 25, 30, 35, 40, 45, 50]
        chpr_accs = [
            0.565789473684211,
            0.717105263157895,
            0.822368421052632,
            0.881578947368421,
            0.914473684210526,
            0.9375,
            0.960526315789474,
        ]
        plt.plot(chpr_thresh, chpr_accs, "mD-", markersize=markersize, label="CHPR")

    else:
        plt.title(
            "auc in [{},{}]: {}".format(thresholds[0], thresholds[-1], auc_all),
            fontsize=40,
        )
    plt.ylim(0, 1)
    plt.xlabel("Error Thresholds (mm)", fontsize=fontsize)
    plt.ylabel("3D PCK", fontsize=fontsize)
    plt.grid(linestyle="-", color="lightgray", alpha=0.5)
    plt.legend(fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.tight_layout()
    plt.savefig(save_pck_file, format="eps")
