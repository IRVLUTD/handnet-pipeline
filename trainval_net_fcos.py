import datetime
import json
import os
import time
from dex_ycb_toolkit.coco_eval import COCOEvaluator
import numpy as np
import torch
import torch.utils.data
import pickle
from tqdm import tqdm, trange
from fcos_utils.fcos import FCOS
from utils.evaluation.evalutils import AverageMeters
from utils.exputils.monitoring import Monitor
from utils.argutils import parse_training_args, parse_general_args
from utils.utils import get_e2e_loaders
from trainval_net_e2e import train_one_epoch_detect
import torchvision.transforms as transforms
import cv2

def visualize_output(images, outputs, targets, image_path):
    images = [ cv2.cvtColor(np.array(transforms.ToPILImage()(image.cpu())), cv2.COLOR_RGB2BGR) for image in images ]
    out_boxes = [ output["boxes"].cpu().numpy() for output in outputs ]
    targ_boxes = [ target["boxes"].numpy() for target in targets ]
    out_labels = [ output["labels"].cpu().numpy() for output in outputs ]
    targ_labels = [ target["labels"].numpy() for target in targets ]

    for i, (image, out_box, targ_box, out_label, targ_label) in enumerate(zip(images, out_boxes, targ_boxes, out_labels, targ_labels)):
        for j, box in enumerate(out_box):
            cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 1)
        for j, box in enumerate(targ_box):
            cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 1)
        cv2.imwrite(image_path, image)

def reshape_output(outputs):

    for output in outputs:
        N = output['boxes'].shape[0]
        output['boxes'] = output['boxes'].reshape(N, 4)
        output['labels'] = output['labels'].reshape(N)
        output['scores'] = output['scores'].reshape(N)
        #output['sides'] = output['sides'].reshape(N)

    return outputs

def evaluate(
    model,
    data_loader,
    args,
    start,
    end,
    display_freq=5000,
    device="cuda",
):
    model.eval()
    time_meters = AverageMeters()

    model_prefix = args.resume[:args.resume.rfind('_') + 1]
    image_path = os.path.join(args.output_dir, 'images/test')
    metric_path = os.path.join(args.output_dir, f'test_metrics/')
    os.makedirs(metric_path, exist_ok=True)

    for epoch in range(end, start - 1, -1):
        if not os.path.exists(model_prefix + f'{epoch}.pth'):
            continue
        checkpoint = torch.load(model_prefix + f'{epoch}.pth', map_location="cpu")

        model.load_state_dict(checkpoint["model"])

        time_meters = AverageMeters()
        pbar = tqdm(data_loader, leave=False)

        all_boxes = []
        image_path_epoch = os.path.join(image_path, f'epoch_{epoch}')
        os.makedirs(image_path_epoch, exist_ok=True)

        with torch.inference_mode():
            for idx, (images, targets) in enumerate(pbar):
                images = [ image.to(device) for image in images ]
                model_time = time.time()
                outputs = model(images)
                model_time = time.time() - model_time
                time_meters.add_loss_value("model_time", model_time)
                pbar.set_postfix({ 
                    'Model Time': time_meters.average_meters['model_time'].avg,
                    'Epoch': epoch,
                })

                outputs = reshape_output(outputs)
                if (idx % display_freq == 0):
                    visualize_output(images, outputs, targets, image_path=os.path.join(image_path_epoch, f'{idx}.png'))

                pred_dict = {}
                for i, output in enumerate(outputs):
                    scores = output["scores"].cpu().numpy()
                    boxes = output["boxes"].cpu().numpy()
                    labels = output["labels"].cpu().numpy()
                    for j, bbox in enumerate(boxes):
                        pred_dict = {}
                        pred_dict["image_id"] = targets[i]["dexycb_id"][0].item()
                        bbox[2:] -= bbox[:2]
                        pred_dict["bbox"] = list(np.float64(bbox))
                        pred_dict["score"] = np.float64(scores[j].item())
                        pred_dict["category_id"] = labels[j].item()
                        all_boxes.append(pred_dict)
        
        print(f"Average fps: {1./time_meters.average_meters['model_time'].avg}")
    
        # pkl results
        with open(os.path.join(metric_path, f'results_{epoch}.pkl'), 'wb+') as f:
            pickle.dump(all_boxes, f)
        with open(os.path.join(metric_path, f'results_{epoch}.json'), 'w+') as f:
            json.dump(all_boxes, f)

    # coco eval
    for epoch in trange(start, end + 1):
        coco_eval = COCOEvaluator('s0_test')
        out_path = os.path.join(metric_path, f'out_{epoch}/')
        results_path = os.path.join(metric_path, f'results_{epoch}.json')
        if not os.path.exists(results_path):
            continue
        os.makedirs(out_path, exist_ok=True)
        coco_eval.evaluate(results_path, out_dir=out_path, tasks=['bbox'])


def main(args):
    device = torch.device(args.device)

    output_dir = args.output_dir

    detect_loader,  detect_test = get_e2e_loaders(args, detect=True)

    print("Creating model")
    #backbone = backbonefpn
    model = FCOS(num_classes=2, ext=False, nms_thresh=0.5)
    model.to(device)

    hosting_folder = os.path.join(args.output_dir, "hosting")
    monitor = Monitor(args.output_dir, hosting_folder=hosting_folder)

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
        evaluate(model, detect_test, args, 1, args.start_epoch - 1, device=device)
        return

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs+1):
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
            if args.amp:
                checkpoint["scaler"] = scaler.state_dict()
            save_name = os.path.join(output_dir, f'detector_{args.session}_{epoch}.pth')
            torch.save(checkpoint, save_name)

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
    print(f"Training time {total_time_str}")

    

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Hand Object Detector FCOS Pretraining on DexYCB')
    parse_general_args(parser)
    parse_training_args(parser)

    args = parser.parse_args()

    args.output_dir = args.save_dir + "/" + args.net + "_" + args.model_name + "_" + str(args.session)
    print(f'\n---------> model output_dir = {args.output_dir}\n')

    os.makedirs(args.output_dir, exist_ok=True)
    
    main(args)
