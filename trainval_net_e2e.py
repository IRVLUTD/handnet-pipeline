# Modify detector in e2e mode:
# In FCOS, pass e2e=True to output feature maps

# Modify 3D branch to use feature map as input (disable base_net)

# Pretrain detector on dexycb detections, train e2e with pretrained detector
# Test detector using detection metrics, test e2e using 3D metricsv 


# Demo and Demo_Video
# Re-implement detection method - Use cv2 version to draw bbox
# Use pytorch3d to render mesh and disable ray

import logging
import math
import sys
import cv2
from parso import parse
from datasets3d.a2jdataset import uvd2xyz, xyz2uvd
from fcos_utils.fcos import FCOS
from trainval_net_mano import prepare_images
from utils.argutils import parse_3d_args, parse_e2e_args
import torch, os, time, datetime
from utils.hpe_eval import hpe_evaluate, hpe_plot
from utils.utils import get_e2e_loaders, vis_minibatch
from e2e_handnet.e2e_handnet import E2EHandNet
from mano_train.evaluation.zimeval import EvalUtil
from progress.bar import Bar as Bar
from mano_train.exputils.monitoring import Monitor
from mano_train.evaluation.evalutils import AverageMeters
from datasets3d.queries import TransQueries
from mano_train.visualize import displaymano
import pickle, numpy as np
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
import torch.nn.functional as F
import torchvision.transforms as Transforms

from utils.vistool import VisualUtil

matplotlib.use('Agg') 

def reshape_output(outputs):

    for output in outputs:
        N = output['boxes'].shape[0]
        output['boxes'] = output['boxes'].reshape(N, 4)
        output['labels'] = output['labels'].reshape(N)
        output['scores'] = output['scores'].reshape(N)
        #output['sides'] = output['sides'].reshape(N)

    return outputs

def train_one_epoch_detect(
    model, 
    optimizer, 
    data_loader, 
    epoch, 
    scaler,
    device,
    args
):
    avg_meters = AverageMeters()
    time_meters = AverageMeters()

    lr_scheduler = None

    if epoch == 1:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )

    pbar = tqdm(data_loader)

    for idx, (images,targets) in enumerate(pbar):
        end = time.time()
        images = list(image.to(device) for image in images)
        targets = list(targets)

        if 'res' in args.net:
            for idx, t in enumerate(targets):
                new_t = {}
                for k, v in t.items():
                    v[v==-1.] = 0.
                    new_t[k] = v.to(device)
                targets[idx] = new_t
        else:
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with torch.cuda.amp.autocast(enabled=scaler is not None):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        loss_value = losses.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict)
            sys.exit(1)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        for key, val in loss_dict.items():
            if val is not None:
                avg_meters.add_loss_value(key, val.item())

        time_meters.add_loss_value("batch_time", time.time() - end)
        avg_meters.add_loss_value("total_loss", loss_value)

        # pbar with loss
        pbar.set_postfix({ 
            'Batch Time': time_meters.average_meters['batch_time'].avg, 
            'Loss' : avg_meters.average_meters['total_loss'].avg,
            'Epoch': epoch
        })

    return avg_meters

def detect_main(args):
    args.det_start_epoch = 1
    device = torch.device(args.device)

    output_dir = args.output_dir

    args.batch_size = args.det_batch_size
    detect_loader, detect_test = get_e2e_loaders(args, detect=True)
    detect_loader.batch_sampler.batch_size = args.mano_batch_size

    print("Creating model")
    #backbone = backbonefpn
    model = FCOS(num_classes=23, ext=False)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]

    if args.det_optimizer == 'sgd':
        optimizer = torch.optim.SGD(
            params, lr=args.det_lr, momentum=args.det_momentum, weight_decay=args.det_weight_decay)
    elif args.det_optimizer == 'adamw':
        optimizer = torch.optim.AdamW(
            params, lr=args.det_lr, weight_decay=args.det_weight_decay)

    scaler = torch.cuda.amp.GradScaler() if args.det_amp else None

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.det_lr_steps, gamma=args.det_lr_gamma)

    hosting_folder = os.path.join(args.output_dir, "hosting")
    monitor = Monitor(args.output_dir, hosting_folder=hosting_folder)

    if args.det_resume:
        checkpoint = torch.load(args.det_resume, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        args.det_start_epoch = checkpoint["epoch"] + 1
        if args.det_amp:
            scaler.load_state_dict(checkpoint["scaler"])

    # if args.test_only:
    #     evaluate(model, detect_test, imdb_test, args, device=device)
    #     return

    print("Start detector training")
    start_time = time.time()
    for epoch in range(args.det_start_epoch, args.det_epochs+1):
        train_avg_meters = train_one_epoch_detect(
            model, 
            optimizer, 
            detect_loader,
            epoch, 
            scaler,
            device,
            args
        )
        lr_scheduler.step()
        if args.output_dir:
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "args": args,
                "epoch": epoch,
            }
            if args.det_amp:
                checkpoint["scaler"] = scaler.state_dict()
            save_name = os.path.join(output_dir, f'detector_{epoch}.pth')
            torch.save(checkpoint, save_name)
            if epoch == args.det_epochs:
                args.pretrained_fcos = save_name

        train_dict = {
            meter_name: meter.avg
            for (
                meter_name,
                meter,
            ) in train_avg_meters.average_meters.items()
        }
        train_full_dict = {**train_dict}
        monitor.log_train(epoch, train_full_dict)

        save_dict = {}
        for key in train_full_dict:
            save_dict[key] = {}
            save_dict[key]["train"] = train_full_dict[key]

        monitor.metrics.save_metrics(epoch, save_dict)
        monitor.metrics.plot_metrics(epoch)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Detector Training time {total_time_str}")

def evaluate(
    model,
    data_loader,
    args,
    epoch,
    device,
    vistool,
    display_freq=5000
):
    logger = logging.getLogger()
    old_level = logger.level
    logger.setLevel(100)
    model.eval()
    image_path = os.path.join(args.output_dir, f'images/test/epoch_{epoch}/')
    os.makedirs(image_path, exist_ok=True)

    time_meters = AverageMeters()
    pbar = tqdm(data_loader, leave=False)

    metric_path = os.path.join(args.output_dir, f'e2e_test_metrics/')
    os.makedirs(metric_path, exist_ok=True)

    with torch.inference_mode():
        for idx, (images, _, sample, depth, paras) in enumerate(pbar):
            images = [ image.to(device) for image in images ]
            depth = depth.to(device)

            model_time = time.time()
            keypoint_pred, depth_im, detections = model(images, depth_images=depth)
            if keypoint_pred == None:
                continue

            model_time = time.time() - model_time
            time_meters.add_loss_value("model_time", model_time)
            pbar.set_postfix({
                'Model Time': time_meters.average_meters['model_time'].avg,
            })

            if (idx % display_freq == 0):
                save_img_path = os.path.join(
                    image_path, "img_{:06d}.png".format(idx)
                )
                color_im = []
                for image, box in zip(images, detections):
                    if box is None:
                        continue
                    color_im_crop = F.interpolate(image[:,  box[1]:box[3] + 1, box[0]:box[2] + 1].unsqueeze(0), size=(176, 176)).squeeze(0)
                    color_im_crop = Transforms.ToPILImage()(color_im_crop).convert('RGB')
                    color_im_crop = cv2.cvtColor(np.array(color_im_crop), cv2.COLOR_BGR2RGB)

                    color_im.append(color_im_crop)
                color_im = np.array(color_im)

                jt_uvd_gt = []
                for jt, para, box in zip(sample[TransQueries.joints3d], paras, detections):
                    if box is not None:
                        jt = jt.numpy()
                        para = para.numpy()
                        box = box.cpu().numpy()
                        X_min, Y_min, X_max, Y_max = box[0], box[1], box[2], box[3]
                        joints_uvd = np.ones((21, 3))
                        joints_uvd[:, 0] = (xyz2uvd(jt, para)[:, 0] - X_min) * 176 / (X_max - X_min)
                        joints_uvd[:, 1] = (xyz2uvd(jt, para)[:, 1] - Y_min) * 176 / (Y_max - Y_min)
                        joints_uvd[:, 2] = xyz2uvd(jt, para)[:, 2]
                        jt_uvd_gt.append(joints_uvd)

                jt_uvd_gt = np.array(jt_uvd_gt).astype(np.float32)

                vis_minibatch(
                    color_im,
                    depth_im.detach().cpu().numpy(),
                    jt_uvd_gt,
                    vistool,
                    sample['dexycb_id'].numpy(),
                    path=save_img_path,
                    jt_pred=keypoint_pred.cpu().numpy(),
                )

            for jt_uvd, box, dexycb_id, para in zip(keypoint_pred, detections, sample['dexycb_id'], paras):
                if box is None:
                    continue
                jt_uvd = jt_uvd.cpu().numpy()
                box = box.cpu().numpy()
                para = para.numpy()
                X_min, Y_min, X_max, Y_max = box[0], box[1], box[2], box[3]
                joints_xyz = np.ones((21, 3))
                joints_xyz[:, 0] = jt_uvd[:, 0] * (X_max - X_min) / 176 + X_min
                joints_xyz[:, 1] = jt_uvd[:, 1] * (Y_max - Y_min) / 176 + Y_min
                joints_xyz[:, 2] = jt_uvd[:, 2]
                joints_xyz = uvd2xyz(joints_xyz, para) * 1000.
            
                j_text = ''
                for j in joints_xyz:
                    j_text += str(list(j)).strip()[1:-1] + ','
                j_text = j_text.replace(" ", "")[:-1]
                
                with open(metric_path + f's0_test_{epoch}.txt', 'a') as output:
                    print(str(dexycb_id.item()) + ',' + j_text, file=output)
    
    print(f"Average fps: {1./time_meters.average_meters['model_time'].avg}")

    hpe_evaluate(args.net, args.output_dir, epoch, epoch)
    #hpe_plot(args.output_dir, epoch, epoch)

    logger.setLevel(old_level)

def e2e_eval_epoch(args):
    device = torch.device(args.device)

    args.batch_size = 1
    _, data_loader_test = get_e2e_loaders(args)

    print("Creating model")
    #backbone = backbonefpn
    model = E2EHandNet(args, reload_detector=True, reload_a2j=True)
    model.to(device)

    vistool = VisualUtil('dexycb')

    evaluate(model, data_loader_test, args, epoch=1, device=device, vistool=vistool)


# training looop
# train FCOS first then train MANO + A2J using same batch size but specific hyperparameters | split dataloaders as well 

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='E2E HandNet Training')
    parse_e2e_args(parser)
    parse_3d_args(parser)

    args = parser.parse_args()

    args.output_dir = args.save_dir + "/" + args.net + "_" + args.model_name
    if 'res' in args.net :
        args.output_dir +=  "_fpn"
    print(f'\n---------> model output_dir = {args.output_dir}\n')

    model_output_dir = args.output_dir

    if args.test_only:
        args.pretrained_fcos = 'models/fcos_dexycb/detector_1_25.pth'
        args.pretrained_a2j = 'models/a2j_dexycb_1/a2j_35.pth'
        e2e_eval_epoch(args)

    args.output_dir = os.path.join(model_output_dir, 'detector')
    os.makedirs(args.output_dir, exist_ok=True)
    detect_main(args)

    args.output_dir = os.path.join(model_output_dir, 'e2e')
    os.makedirs(args.output_dir, exist_ok=True)
    mano_main(args)