import os
import pickle
import time

import numpy as np
from progress.bar import Bar as Bar
import torch

from datasets3d.queries import BaseQueries, TransQueries

from mano_train.evaluation.evalutils import AverageMeters
from mano_train.evaluation.zimeval import EvalUtil
from mano_train.visualize import displaymano
from mano_train.netscripts import savemano
from manopth.manolayer import ManoLayer
import torch.nn.functional as torch_f


def save_metrics(sample, results, metric_path, mano_layer_left, mano_layer_right, epoch, joints2d):
    os.makedirs(metric_path, exist_ok=True)
    for i in range(len(results['joints'])):
        #pose = results['pose'][i]
        #betas = results['shape'][i]
        label_file = sample['label_file'][i]
        #side = sample[BaseQueries.sides][i]
        label = np.load(label_file)
        joints3d = results['joints'][i]

        pose_m = label['pose_m']
        #pose_d = torch.from_numpy(pose_m).cuda()

        # if str(side.numpy()) == '0':
        #     verts2, joints3d = mano_layer_left(pose.unsqueeze(0), betas.unsqueeze(0).cuda(), pose_d[:, 48:51])
        # else:
        #     verts2, joints3d = mano_layer_right(pose.unsqueeze(0), betas.unsqueeze(0).cuda(), pose_d[:, 48:51])
        
        joints3d = joints3d.cpu().detach().numpy().squeeze().squeeze()
        j_text = ''
        for j in joints3d:
            j_text += str(list(j)).strip()[1:-1] + ','
        j_text = j_text.replace(" ", "")[:-1]
        
        with open(metric_path + f's0_test_{epoch}.txt', 'a') as output:
            print(str(sample['idx'][i].numpy()) + ',' + j_text, file=output)
    
    if joints2d:
        p_joints2d = results['joints2d']
        p_joints2d = { str(sample['idx'][idx].numpy()): i.cpu().detach().numpy().squeeze().squeeze() for idx, i in enumerate(results['joints2d']) }
        joints2d = {}

        for idx, jnt2d in p_joints2d.items():
            root_out = (jnt2d[0] < 0).sum() + (jnt2d[1] < 0).sum() + (jnt2d[0] > 256).sum() + (jnt2d[1] > 256).sum()
            if root_out == 0:
                joints2d[idx] = jnt2d
           

        jnt2_dir = os.path.join(metric_path, f'jnt2_{epoch}/')
        os.makedirs(jnt2_dir, exist_ok=True)
        iternum = 1
        while os.path.exists(os.path.join(jnt2_dir, f'{str(iternum)}.pkl') ):
            iternum+=1
        with open( os.path.join(jnt2_dir, f'{str(iternum)}.pkl'), 'wb+') as output:
            pickle.dump(joints2d, output)

def th_delete(tensor, indices):
    mask = torch.ones(tensor.numel(), dtype=torch.bool)
    mask[indices] = False
    return tensor[mask]

def epoch_pass(
    loader,
    model,
    epoch,
    optimizer=None,
    freeze_batchnorm=False,
    display=True,
    display_freq=10,
    save_path="checkpoints/debug",
    idxs=None,
    train=True,
    inspect_weights=False,
    fig=None,
    metrics=None,
    return_joints2d=False,
):
    avg_meters = AverageMeters()
    time_meters = AverageMeters()
    print("epoch: {}".format(epoch))

    idxs = list(range(21))  # Joints to use for evaluation
    evaluator = EvalUtil()

    with open("misc/mano/models/MANO_RIGHT.pkl", "rb") as p_f:
        mano_right_data = pickle.load(p_f, encoding="latin1")
        faces_right = mano_right_data["f"]
    with open("misc/mano/models/MANO_LEFT.pkl", "rb") as p_f:
        mano_left_data = pickle.load(p_f, encoding="latin1")
        faces_left = mano_left_data["f"]

    # Switch to correct model mode
    if train:
        if freeze_batchnorm:
            model.eval()
        else:
            model.train()

        save_img_folder = os.path.join(
            save_path, "images", "train", "epoch_{}".format(epoch)
        )
    else:
        model.eval()
        save_img_folder = os.path.join(
            save_path, "images", "val", "epoch_{}".format(epoch)
        )
    os.makedirs(save_img_folder, exist_ok=True)

    end = time.time()
    bar = Bar("Processing", max=len(loader))
    lim = 0

    mano_layer_left = ManoLayer(ncomps=45,
                        flat_hand_mean=False,
                        #center_idx=9,
                        side='left',
                        mano_root='/home/cgalab/handobj/manopth/mano/models',
                        use_pca=True).cuda()

    mano_layer_right = ManoLayer(ncomps=45,
                        flat_hand_mean=False,
                        #center_idx=9,
                        side='right',
                        mano_root='/home/cgalab/handobj/manopth/mano/models',
                        use_pca=True).cuda()

    for batch_idx, (sample) in enumerate(loader):
        # if lim == 1:
        #     break
        if "vis" in sample:
            visibilities = sample["vis"].numpy()
        else:
            visibilities = None
        # measure data loading time
        deleted = 0

        for idx, i in enumerate(sample[TransQueries.joints2d]):
            j = idx - deleted
            joints2d_np = i.numpy()
            x_out = (joints2d_np[:, 0] < 0).sum() + (joints2d_np[:, 0] > 256).sum()
            y_out = (joints2d_np[:, 1] < 0).sum() + (joints2d_np[:, 1] > 256).sum()
            root_out = (joints2d_np[0, 0] < 0).sum() + (joints2d_np[0, 0] > 256).sum() + (joints2d_np[0, 1] < 0).sum() + (joints2d_np[0, 1] > 256).sum()
            # exception = True if train==False and sample['idx'][j].item() == 32000 else False
            if not joints2d_np.any() or x_out > 12 or y_out > 12 or root_out > 0: 
                sample[TransQueries.joints3d] =  torch.cat([sample[TransQueries.joints3d][:j], sample[TransQueries.joints3d][j+1:]])
                if TransQueries.verts3d in sample:
                    sample[TransQueries.verts3d] =  torch.cat([sample[TransQueries.verts3d][:j], sample[TransQueries.verts3d][j+1:]])
                sample[TransQueries.images] = torch.cat([sample[TransQueries.images][:j], sample[TransQueries.images][j+1:]])
                sample[BaseQueries.sides] = sample[BaseQueries.sides][:j] + sample[BaseQueries.sides][j+1:]
                # sample[TransQueries.center3d] = torch.cat([sample[TransQueries.center3d][:j], sample[TransQueries.center3d][j+1:]])
                sample[TransQueries.joints2d] = torch.cat([sample[TransQueries.joints2d][:j], sample[TransQueries.joints2d][j+1:]])
                # sample['pose'] = torch.cat([sample['pose'][:j], sample['pose'][j+1:]])
                sample['idx'] = torch.cat([sample['idx'][:j], sample['idx'][j+1:]])
                sample['label_file'] = sample['label_file'][:j] + sample['label_file'][j+1:]
                # sample[BaseQueries.images] = sample[BaseQueries.images][:j] + sample[BaseQueries.images][j+1:]

                #sample['joints2d'] = torch.cat([sample['joints2d'][:j], sample['joints2d'][j+1:]])
                # sample['trans'] = torch.cat([sample['trans'][:j], sample['trans'][j+1:]])
                
                deleted += 1


        sample[BaseQueries.sides] = torch.from_numpy(np.array([ 0 if i == 'left' else 1 for i in sample[BaseQueries.sides]]))

        # if train:
        #     sample_pkl = open('pkl_cache/sample.pkl', 'ab')
        #     pickle.dump(sample, sample_pkl)
        #     sample_pkl.close()
        time_meters.add_loss_value("data_time", time.time() - end)

        # Compute output
        _, results, _ = model.forward(
            sample, return_features=inspect_weights, no_loss=True
        )

        model_losses = {}
        model_loss = None

        results_names_dict = {
            "verts": TransQueries.verts3d,
            "joints": TransQueries.joints3d,
            "joints2d": TransQueries.joints2d,
        }

        for r_name in results_names_dict.keys():
            gt = sample[results_names_dict[r_name]].cuda()
            model_losses[r_name] = torch_f.mse_loss(
                results[r_name], gt.float()
            ).float()

        model_loss = 0.167*sum(model_losses.values())
        model_losses['total_loss'] = model_loss

        # compute gradient and do SGD step
        if train:
            optimizer.zero_grad()
            if inspect_weights:
                model_loss.backward(retain_graph=True)
            else:
                model_loss.backward()
            optimizer.step()
            if inspect_weights:
                inspect_loss_names = [
                    "mano_verts3d",
                    "mano_shape",
                ]
                features = results["img_features"]
                features.retain_grad()
                for inspect_loss_name in inspect_loss_names:
                    features.grad = None
                    if inspect_loss_name in model_losses:
                        loss_val = model_losses[inspect_loss_name]
                        if loss_val is not None:
                            loss_val.backward(retain_graph=True)
                            print(inspect_loss_name, torch.norm(features.grad).item())

        # Get values out of tensors
        for loss in model_losses:
            if model_losses[loss] is not None:
                tensor = model_losses[loss]
                value = model_losses[loss].item()
                model_losses[loss] = value
                if value > 100000:
                    print(loss, tensor, model_losses[loss])

        for key, val in model_losses.items():
            if val is not None:
                avg_meters.add_loss_value(key, val)

        save_img_path = os.path.join(
            save_img_folder, "img_{:06d}.png".format(batch_idx)
        )
        if (batch_idx % display_freq == 0) and display:
            displaymano.visualize_batch(
                save_img_path,
                fig=fig,
                sample=sample,
                results=results,
                faces_right=faces_right,
                faces_left=faces_left,
            )

        if "joints" in results and TransQueries.joints3d in sample:
            preds = results["joints"].detach().cpu()
            # Keep only evaluation joints
            preds = preds[:, idxs]
            gt = sample[TransQueries.joints3d][:, idxs]

            # Feed predictions to evaluator
            if visibilities is None:
                visibilities = [None] * len(gt)
            for gt_kp, pred_kp, visibility in zip(gt, preds, visibilities):
                evaluator.feed(gt_kp, pred_kp, keypoint_vis=visibility)

        if metrics:
            save_metrics(sample, results, metrics, mano_layer_left, mano_layer_right, epoch, joints2d=return_joints2d)

        # measure elapsed time
        time_meters.add_loss_value("batch_time", time.time() - end)

        # plot progress
        bar.suffix = "({batch}/{size}) Data: {data:.6f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f}".format(
        batch=batch_idx + 1,
        size=len(loader),
        data=time_meters.average_meters["data_time"].val,
        bt=time_meters.average_meters["batch_time"].avg,
        total=bar.elapsed_td,
        eta=bar.eta_td,
        loss=avg_meters.average_meters["total_loss"].avg,
        )
        bar.next()
        lim += 1

    (
        epe_mean_all,
        _,
        epe_median_all,
        auc_all,
        pck_curve_all,
        thresholds,
    ) = evaluator.get_measures(0, 50, 20)
    if "joints" in results:
        if train:
            pck_folder = os.path.join(save_path, "pcks/train")
        else:
            pck_folder = os.path.join(save_path, "pcks/val")
        os.makedirs(pck_folder, exist_ok=True)
        pck_info = {
            "auc": auc_all,
            "thres": thresholds,
            "pck_curve": pck_curve_all,
            "epe_mean": epe_mean_all,
            "epe_median": epe_median_all,
            "evaluator": evaluator,
        }

        save_pck_file = os.path.join(pck_folder, "epoch_{}.eps".format(epoch))
        if sample["dataset"] == "stereohands" and (sample["split"] == "test"):
            overlay = "stereo_test"
        elif sample["dataset"] == "stereohands" and (sample["split"] == "all"):
            overlay = "stereo_all"
        else:
            overlay = None

        if np.isnan(auc_all):
            print(
                "Not saving pck info, normal in case of only 2D info supervision, abnormal otherwise"
            )
        else:
            displaymano.save_pck_img(
                thresholds, pck_curve_all, auc_all, save_pck_file, overlay=overlay
            )
        save_pck_pkl = os.path.join(pck_folder, "epoch_{}.pkl".format(epoch))
        with open(save_pck_pkl, "wb") as p_f:
            pickle.dump(pck_info, p_f)

    else:
        pck_info = {}

    bar.finish()
    return avg_meters, pck_info
