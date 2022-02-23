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
from utils.utils import get_e2e_loaders

from trainval_net_fpn import train_one_epoch


def main(args):
    device = torch.device(args.device)

    output_dir = args.output_dir

    detect_loader,  detect_test = get_e2e_loaders(args, detect=True)

    print("Creating model")
    #backbone = backbonefpn
    model = FCOS(num_classes=23, ext=False)
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

    # if args.test_only:
    #     evaluate(model, data_loader_test, imdb_test, args, device=device)
    #     return

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs+1):
        train_one_epoch(model, optimizer, detect_loader, device, epoch, args.print_freq, args, scaler)
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
    parser = argparse.ArgumentParser(description='Hand Object Detector FCOS Pretraining on DexYCB')
    parse_general_args(parser)
    parse_detection_args(parser)

    args = parser.parse_args()

    args.output_dir = args.save_dir + "/" + args.net + "_" + args.model_name
    print(f'\n---------> model output_dir = {args.output_dir}\n')

    os.makedirs(args.output_dir, exist_ok=True)
    
    main(args)
