import datetime
import os
import time

import torch
import torch.utils.data
from torch import log_, nn
import torchvision
import torchvision.models.detection


import pickle

from fpn_utils.faster_rcnn_fpn import FasterRCNN

from model.utils.net_utils import save_checkpoint
import fpn_utils.utils as utils
import math, sys

from fcos_utils.fcos import FCOS

from utils.argutils import parse_detection_args, parse_general_args
from utils.utils import get_loaders_100doh
from tqdm import tqdm

import numpy as np

def collate_fn(batch):
    return tuple(zip(*batch))

def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq, args, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )
    #for idx, (images, targets) in enumerate(tqdm(data_loader)):
    for idx, (images, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
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

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
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

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return metric_logger

def reshape_output(outputs):

    for output in outputs:
        N = output['boxes'].shape[0]
        output['boxes'] = output['boxes'].reshape(N, 4)
        output['labels'] = output['labels'].reshape(N, 1)
        output['scores'] = output['scores'].reshape(N, 1)
        output['contacts'] = output['contacts'].reshape(N, 1)
        output['dxdymags'] = output['dxdymags'].reshape(N, 3)
        output['sides'] = output['sides'].reshape(N, 1)

    return outputs

def evaluate(model, data_loader, imdb, args, device):
    if args.net != 'fcos':
        output_dir = os.path.join('output/', args.net, imdb.name, args.model_name + '_fpn')
    else:
        output_dir = os.path.join('output/', args.net, imdb.name, args.model_name)
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"
    all_boxes = [[[] for _ in range(len(data_loader.dataset))] for _ in range(imdb.num_classes)]
    empty_array = np.transpose(np.array([[],[],[],[],[]]), (1,0))

    with torch.no_grad():
        for images, targets in metric_logger.log_every(data_loader, 100, header):
            images = list(img.to(device) for img in images)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            model_time = time.time()
            outputs = model(images)
            model_time = time.time() - model_time

            obj_ind = [torch.nonzero( (t['scores'] > 0.1) & (t['labels'] == 1) ).squeeze() for t in outputs]
            hand_ind = [torch.nonzero( (t['scores'] > 0.1) & (t['labels'] == 2) ).squeeze() for t in outputs]

            outputs = reshape_output(outputs)
            
            obj_final = [
               torch.cat((
                    t['boxes'][i],
                    t['scores'][i],
                    t['contacts'][i],
                    t['dxdymags'][i],
                    t['sides'][i],
                    torch.ones_like(t['sides'][i]) # filler for nc_prob
                ), -1).reshape(-1, 11).cpu().numpy()
                for t, i in zip(outputs, obj_ind)
            ]
            hand_final = [
                torch.cat((
                    t['boxes'][i],
                    t['scores'][i],
                    t['contacts'][i],
                    t['dxdymags'][i],
                    t['sides'][i],
                    torch.ones_like(t['sides'][i]) # filler for nc_prob
                ), -1).reshape(-1, 11).cpu().numpy()
                for t, i in zip(outputs, hand_ind)
            ]
            ids = [t['image_id'].item() for t in targets]
            
            for idx, id in enumerate(ids):
                if obj_final[idx].size == 0:
                    all_boxes[1][id] = empty_array
                if hand_final[idx].size == 0:
                    all_boxes[2][id] = empty_array
                
                else:
                    all_boxes[1][id] = obj_final[idx]
                    all_boxes[2][id] = hand_final[idx]

            metric_logger.update(model_time=model_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    imdb.evaluate_detections(all_boxes, output_dir)
    print("Averaged stats:", metric_logger)
    print("FPS:", 1.0 / float(str(metric_logger)[12:18]))

def main(args):
    device = torch.device(args.device)

    output_dir = args.output_dir

    data_loader, data_loader_test, imdb, imdb_test, num_classes = get_loaders_100doh(args)

    print("Creating model")
    #backbone = backbonefpn
    if args.net == 'fcos':
        model = FCOS(num_classes=num_classes)
    else:
        model = FasterRCNN(num_classes=num_classes, num_layers=int(args.net[3:]))
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]

    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(
            params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(
            params, lr=args.lr, weight_decay=args.weight_decay)

    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_steps, gamma=args.lr_gamma)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        args.start_epoch = checkpoint["epoch"] + 1
        if args.amp:
            scaler.load_state_dict(checkpoint["scaler"])

    if args.test_only:
        evaluate(model, data_loader_test, imdb_test, args, device=device)
        return


    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs+1):
        train_one_epoch(model, optimizer, data_loader, device, epoch, args.print_freq, args, scaler)
        lr_scheduler.step()
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
            save_name = os.path.join(output_dir, f'detector_{args.session}_{epoch}.pth')
            save_checkpoint(checkpoint, save_name)

        #evaluate after every epoch
        #evaluate(model, data_loader_test, device=device)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")

    

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Hand Object Detector with FPN (All ResNets) or FCOS')
    parse_general_args(parser)
    parse_detection_args(parser)

    args = parser.parse_args()

    args.output_dir = args.save_dir + "/" + args.net + "_" + args.model_name
    if 'res' in args.net :
        args.output_dir +=  "_fpn"
    print(f'\n---------> model output_dir = {args.output_dir}\n')

    os.makedirs(args.output_dir, exist_ok=True)
    
    main(args)
