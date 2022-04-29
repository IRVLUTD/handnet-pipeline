# Modify detector in e2e mode:
# In FCOS, pass e2e=True to output feature maps

# Modify 3D branch to use feature map as input (disable base_net)

# Pretrain detector on dexycb detections, train e2e with pretrained detector
# Test detector using detection metrics, test e2e using 3D metricsv 


# Demo and Demo_Video
# Re-implement detection method - Use cv2 version to draw bbox
# Use pytorch3d to render mesh and disable ray

import sys

from tqdm import tqdm
from utils.argutils import parse_training_args, parse_general_args, parse_a2j_args
import torch, os, time, datetime
from utils.utils import get_e2e_loaders, vis_minibatch
from utils.evaluation.zimeval import EvalUtil
from progress.bar import Bar as Bar
from utils.exputils.monitoring import Monitor
from utils.evaluation.evalutils import AverageMeters
from utils.visualize import displaymano
import pickle, numpy as np
import matplotlib
from utils.hpe_eval import hpe_evaluate, hpe_plot
import logging
from utils.vistool import VisualUtil
from a2j.a2j import A2JModel

matplotlib.use('Agg') 

from datasets3d.a2jdataset import uvd2xyz

def convert_joints(jt_uvd_pred, jt_uvd_gt, box, paras, cropWidth, cropHeight):
    jt_uvd_pred = jt_uvd_pred.reshape(-1, 3)
    jt_uvd_gt = jt_uvd_gt.reshape(-1, 3)
    box = box.reshape(4)
    paras = paras.reshape(4)

    X_min, Y_min, X_max, Y_max = box[0], box[1], box[2], box[3]

    jt_xyz_pred = np.ones_like(jt_uvd_pred)
    jt_xyz_pred[:, 0] = jt_uvd_pred[:, 0] * (X_max - X_min) / cropWidth + X_min
    jt_xyz_pred[:, 1] = jt_uvd_pred[:, 1] * (Y_max - Y_min) / cropHeight + Y_min
    jt_xyz_pred[:, 2] = jt_uvd_pred[:, 2]
    jt_xyz_pred = uvd2xyz(jt_xyz_pred, paras) * 1000.

    jt_xyz_gt = np.ones_like(jt_uvd_gt)
    jt_xyz_gt[:, 0] = jt_uvd_gt[:, 0] * (X_max - X_min) / cropWidth + X_min
    jt_xyz_gt[:, 1] = jt_uvd_gt[:, 1] * (Y_max - Y_min) / cropHeight + Y_min
    jt_xyz_gt[:, 2] = jt_uvd_gt[:, 2]
    jt_xyz_gt = uvd2xyz(jt_xyz_gt, paras) * 1000.

    return jt_xyz_pred, jt_xyz_gt

def evaluate(
    model,
    data_loader,
    scaler,
    device,
    vistool,
    args,
    start,
    end,
    display_freq=5000
):
    logger = logging.getLogger()
    old_level = logger.level
    logger.setLevel(100)
    model.eval()
    image_path = os.path.join(args.output_dir, 'images/test')
    os.makedirs(image_path, exist_ok=True)

    model_prefix = args.resume[:args.resume.rfind('_') + 1]


    for epoch in range(start, end + 1):
        checkpoint = torch.load(model_prefix + f'{epoch}.pth', map_location="cpu")
        model.load_state_dict(checkpoint["model"])

        time_meters = AverageMeters()
        pbar = tqdm(data_loader, leave=False)

        metric_path = os.path.join(args.output_dir, f'a2j_test_metrics/')
        os.makedirs(metric_path, exist_ok=True)
        image_path_epoch = os.path.join(image_path, f'epoch_{epoch}')
        os.makedirs(image_path_epoch, exist_ok=True)

        with torch.inference_mode():
            for idx, (im, jt_uvd_gt, dexycb_id, color_im, box, paras) in enumerate(pbar):
                im = im.to(device)
                model_time = time.time()
                with torch.cuda.amp.autocast(enabled=scaler is not None):
                    keypoint_pred = model(im)

                jt_xyz_pred, jt_xyz_gt = convert_joints(
                    keypoint_pred.detach().cpu().numpy(), 
                    jt_uvd_gt.numpy(), 
                    box.numpy(), 
                    paras.numpy(), 
                    data_loader.dataset.cropWidth, 
                    data_loader.dataset.cropHeight
                )
                model_time = time.time() - model_time
                time_meters.add_loss_value("model_time", model_time)
                pbar.set_postfix({
                    'Model Time': time_meters.average_meters['model_time'].avg,
                    'Epoch': epoch,
                })

                if (idx % display_freq == 0):
                    save_img_path = os.path.join(
                        image_path_epoch, "img_{:06d}.png".format(idx)
                    )
                    vis_minibatch(
                        color_im.numpy(),
                        im.detach().cpu().numpy(),
                        jt_uvd_gt.numpy(),
                        vistool,
                        dexycb_id.numpy(),
                        path=save_img_path,
                        jt_pred=keypoint_pred.cpu().numpy(),
                    )

                j_text = ''
                for j in jt_xyz_pred:
                    j_text += str(list(j)).strip()[1:-1] + ','
                j_text = j_text.replace(" ", "")[:-1]
                
                with open(metric_path + f's0_test_{epoch}.txt', 'a') as output:
                    print(str(dexycb_id[0].numpy())[1:-1] + ',' + j_text, file=output)
        
        print(f"Average fps: {1./time_meters.average_meters['model_time'].avg}")

    hpe_evaluate(args.net, args.output_dir, start, end)
    hpe_plot(args.output_dir, start, end)

    logger.setLevel(old_level)
            

def train_one_epoch_a2j(
    model, 
    optimizer, 
    data_loader, 
    epoch, 
    scaler,
    device,
    vistool,
    display_freq=5000
):
    avg_meters = AverageMeters()
    time_meters = AverageMeters()
    evaluator = EvalUtil()

    save_img_folder = os.path.join(
           args.output_dir , "images", "train", "epoch_{}".format(epoch)
    )

    os.makedirs(save_img_folder, exist_ok=True)
    pbar = tqdm(data_loader)
    idxs = list(range(21))

    for idx, (im, jt_uvd_gt, dexycb_id, color_im, _, _) in enumerate(pbar):
        # visualize batch
        # vis_minibatch(
        #     color_im.detach().cpu().numpy(),
        #     im.detach().cpu().numpy(),
        #     jt_uvd_gt.detach().cpu().numpy(),
        #     vistool
        # )

        end = time.time()
        im = im.to(device)
        jt_uvd_gt = jt_uvd_gt.to(device)
        time_meters.add_loss_value("data_time", time.time() - end)

        with torch.cuda.amp.autocast(enabled=scaler is not None):
            # model_losses, results, target = model(images, sample)
            model_losses, keypoint_pred = model(im, jt_uvd_gt)
        model_loss = model_losses['total_loss']

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(model_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            model_loss.backward()
            optimizer.step()

        for loss in model_losses:
            if model_losses[loss] is not None:
                value = model_losses[loss].item()
                model_losses[loss] = value

        for key, val in model_losses.items():
            if val is not None:
                avg_meters.add_loss_value(key, val)

        save_img_path = os.path.join(
            save_img_folder, "img_{:06d}.png".format(idx)
        )
        if (idx % display_freq == 0):
            vis_minibatch(
                color_im.detach().cpu().numpy(),
                im.detach().cpu().numpy(),
                jt_uvd_gt.detach().cpu().numpy(),
                vistool,
                dexycb_id.detach().cpu().numpy(),
                path=save_img_path,
                jt_pred=keypoint_pred.detach().cpu().numpy()
            )

        preds = keypoint_pred.detach().cpu()
        # Keep only evaluation joints
        preds = preds[:, idxs]
        gt = jt_uvd_gt.detach().cpu()[:, idxs]

        # Feed predictions to evaluator
        visibilities = [None] * len(gt)
        for gt_kp, pred_kp, visibility in zip(gt, preds, visibilities):
            evaluator.feed(gt_kp, pred_kp, keypoint_vis=visibility)

        time_meters.add_loss_value("batch_time", time.time() - end)

        # pbar with loss
        pbar.set_postfix({ 
            'Batch Time': time_meters.average_meters['batch_time'].avg, 
            'Loss' : avg_meters.average_meters['total_loss'].avg,
            'Epoch': epoch,
        })

    (
        epe_mean_all,
        _,
        epe_median_all,
        auc_all,
        pck_curve_all,
        thresholds,
    ) = evaluator.get_measures(0, 50, 20)
    pck_folder = os.path.join(args.output_dir, "pcks/train")
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

    return avg_meters, pck_info
        


def main(args):
    device = torch.device(args.device)

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    data_loader, data_loader_test = get_e2e_loaders(args, a2j=True)
    data_loader.batch_sampler.batch_size = args.batch_size
    data_loader.num_workers = args.workers

    # Load model
    model = A2JModel(num_classes=21, crop_height=data_loader.dataset.cropHeight, crop_width=data_loader.dataset.cropWidth)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(
            params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(
            params, lr=args.lr, weight_decay=args.weight_decay)


    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    #lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_steps, gamma=args.lr_gamma)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    vistool = VisualUtil('dexycb')

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        args.start_epoch = checkpoint["epoch"] + 1
        if args.amp:
            scaler.load_state_dict(checkpoint["scaler"])

    if args.test_only:
        evaluate(model, data_loader_test, scaler, device, vistool, args, 1, args.start_epoch - 1)
        return


    print("Start training")
    start_time = time.time()

    hosting_folder = os.path.join(args.output_dir, "hosting")
    monitor = Monitor(args.output_dir, hosting_folder=hosting_folder)

    for epoch in range(args.start_epoch, args.epochs+1):
        train_avg_meters, train_pck_infos = train_one_epoch_a2j(
            model, 
            optimizer, 
            data_loader,
            epoch, 
            scaler,
            device,
            vistool
        )
        lr_scheduler.step()

        # Save custom logs
        train_dict = {
            meter_name: meter.avg
            for (
                meter_name,
                meter,
            ) in train_avg_meters.average_meters.items()
        }
        if train_pck_infos:
            train_pck_dict = {
                "auc": train_pck_infos["auc"],
                "epe_mean": train_pck_infos["epe_mean"],
                "epe_median": train_pck_infos["epe_median"],
            }
        else:
            train_pck_dict = {}
        train_full_dict = {**train_dict, **train_pck_dict}
        monitor.log_train(epoch, train_full_dict)

        save_dict = {}
        for key in train_full_dict:
            save_dict[key] = {}
            save_dict[key]["train"] = train_full_dict[key]

        monitor.metrics.save_metrics(epoch, save_dict)
        monitor.metrics.plot_metrics(epoch)

        if args.output_dir:
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "args": args,
                "epoch": epoch,
            }
            if args.amp:
                checkpoint["scaler"] = scaler.state_dict()
            save_name = os.path.join(output_dir, f'{args.net}_{epoch}.pth')
            torch.save(checkpoint, save_name)

        #evaluate after every epoch
        #evaluate(model, data_loader_test, device=device)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")

    

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='E2E A2J Training')
    parse_general_args(parser)
    parse_a2j_args(parser)
    parse_training_args(parser)
    args = parser.parse_args()

    args.output_dir = args.save_dir + "/" + args.net + "_" + args.model_name + "_" + str(args.session)
    if 'a2j' not in args.net:
        sys.exit("Only a2j network is supported")
    print(f'\n---------> model output_dir = {args.output_dir}\n')

    args.aspect_ratio_group_factor = 0

    os.makedirs(args.output_dir, exist_ok=True)
    main(args)