# Modify detector in e2e mode:
# In FCOS, pass e2e=True to output feature maps

# Modify 3D branch to use feature map as input (disable base_net)

# Pretrain detector on dexycb detections, train e2e with pretrained detector
# Test detector using detection metrics, test e2e using 3D metricsv 


# Demo and Demo_Video
# Re-implement detection method - Use cv2 version to draw bbox
# Use pytorch3d to render mesh and disable ray

from parso import parse
from utils.argutils import parse_3d_args, parse_detection_args, parse_general_args
import torch, os, time, datetime
from utils.utils import get_e2e_loaders
from e2e_handnet.e2e_handnet import E2EHandNet
from mano_train.evaluation.zimeval import EvalUtil
from progress.bar import Bar as Bar
from mano_train.exputils.monitoring import Monitor
from mano_train.evaluation.evalutils import AverageMeters
from datasets3d.queries import TransQueries
from mano_train.visualize import displaymano
import pickle, numpy as np
import json

def reshape_output(outputs):

    for output in outputs:
        N = output['boxes'].shape[0]
        output['boxes'] = output['boxes'].reshape(N, 4)
        output['labels'] = output['labels'].reshape(N)
        output['scores'] = output['scores'].reshape(N)
        #output['sides'] = output['sides'].reshape(N)

    return outputs

def evaluate(model, data_loader_e2e, data_loader_detect, args):
    model.eval()

    all_boxes = []

    with torch.no_grad():
        for batch_idx, (images, targets, sample) in enumerate(zip(data_loader_detect, data_loader_e2e)):
            images = [ image.cuda() for image in images ]
            with torch.inference_mode():
                results, detections, batch_idx, level_idx, boxes = model(images, sample)
            
            outputs = detections

            outputs = reshape_output(outputs)

            pred_dict = {}
            for i, output in enumerate(outputs):
                pred_dict["image_id"] = targets[i]["image_id"].item()
                scores = output["scores"].cpu().numpy()
                boxes = output["boxes"].cpu().numpy()
                labels = output["labels"].cpu().numpy()
                for j, bbox in enumerate(boxes):
                    bbox[2:] -= bbox[:2]
                    pred_dict["bbox"] = bbox
                    pred_dict["score"] = scores[j].item()
                    pred_dict["category_id"] = labels[j].item()
                    all_boxes.append(pred_dict)
    
    with open('results.json', 'w+') as f:
        json.dump(all_boxes, f)

    return all_boxes
            

def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq, scaler):
    avg_meters = AverageMeters()
    time_meters = AverageMeters()
    evaluator = EvalUtil()
    print("epoch: {}".format(epoch))

    save_img_folder = os.path.join(
           args.output_dir , "images", "train", "epoch_{}".format(epoch)
    )

    os.makedirs(save_img_folder, exist_ok=True)
    bar = Bar("Processing", max=len(data_loader))
    idxs = list(range(21))

    for idx, (images, target) in enumerate(data_loader):
        end = time.time()
        images = list(img.cuda() for img in images)
        time_meters.add_loss_value("data_time", time.time() - end)

        with torch.cuda.amp.autocast(enabled=scaler is not None):
            # model_losses, results, target = model(images, sample)
            model_losses, results, target = model(images, target)
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
                tensor = model_losses[loss]
                value = model_losses[loss].item()
                model_losses[loss] = value

        for key, val in model_losses.items():
            if val is not None:
                avg_meters.add_loss_value(key, val)

        save_img_path = os.path.join(
            save_img_folder, "img_{:06d}.png".format(idx)
        )
        # if (idx % display_freq == 0) and display:
        #     displaymano.visualize_batch(
        #         save_img_path,
        #         fig=fig,
        #         sample=sample,
        #         results=results,
        #         faces_right=faces_right,
        #         faces_left=faces_left,
        #     )

        if "joints" in results and TransQueries.joints3d in target:
            preds = results["joints"].detach().cpu()
            # Keep only evaluation joints
            preds = preds[:, idxs]
            gt = target[TransQueries.joints3d][:, idxs]

            # Feed predictions to evaluator
            visibilities = [None] * len(gt)
            for gt_kp, pred_kp, visibility in zip(gt, preds, visibilities):
                evaluator.feed(gt_kp, pred_kp, keypoint_vis=visibility)

        time_meters.add_loss_value("batch_time", time.time() - end)

        # plot progress
        bar.suffix = "({batch}/{size}) Data: {data:.6f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f}".format(
        batch=idx + 1,
        size=len(data_loader),
        data=time_meters.average_meters["data_time"].val,
        bt=time_meters.average_meters["batch_time"].avg,
        total=bar.elapsed_td,
        eta=bar.eta_td,
        loss=avg_meters.average_meters["total_loss"].avg,
        )
        bar.next()

    (
        epe_mean_all,
        _,
        epe_median_all,
        auc_all,
        pck_curve_all,
        thresholds,
    ) = evaluator.get_measures(0, 50, 20)
    if "joints" in results:
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

    else:
        pck_info = {}

    bar.finish()
    return avg_meters, pck_info
        


def main(args):
    device = torch.device(args.device)

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    data_loader,  data_loader_test = get_e2e_loaders(args)
    data_loader.batch_sampler.batch_size = args.batch_size
    data_loader.num_workers = args.workers

    # Load model
    model = E2EHandNet(args)
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
        _, detect_data_loader_test = get_e2e_loaders(args, detect=True) 
        evaluate(model, data_loader_test, detect_data_loader_test, args)
        return


    print("Start training")
    start_time = time.time()

    hosting_folder = os.path.join(args.output_dir, "hosting")
    monitor = Monitor(args.output_dir, hosting_folder=hosting_folder)

    for epoch in range(args.start_epoch, args.epochs+1):
        train_avg_meters, train_pck_infos = train_one_epoch(model, optimizer, data_loader, device, epoch, args.print_freq, scaler)
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
            save_name = os.path.join(output_dir, f'e2e_handnet_{args.session}_{epoch}.pth')
            torch.save(checkpoint, save_name)

        #evaluate after every epoch
        #evaluate(model, data_loader_test, device=device)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")

    

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='E2E HandNet Training')
    parse_general_args(parser)
    parse_detection_args(parser)
    parse_3d_args(parser)

    args = parser.parse_args()

    args.output_dir = args.save_dir + "/" + args.net + "_" + args.model_name
    if 'res' in args.net :
        args.output_dir +=  "_fpn"
    print(f'\n---------> model output_dir = {args.output_dir}\n')

    os.makedirs(args.output_dir, exist_ok=True)
     
    main(args)