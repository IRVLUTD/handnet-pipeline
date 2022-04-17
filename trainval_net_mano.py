# Modify detector in e2e mode:
# In FCOS, pass e2e=True to output feature maps

# Modify 3D branch to use feature map as input (disable base_net)

# Pretrain detector on dexycb detections, train e2e with pretrained detector
# Test detector using detection metrics, test e2e using 3D metricsv 


# Demo and Demo_Video
# Re-implement detection method - Use cv2 version to draw bbox
# Use pytorch3d to render mesh and disable ray

from parso import parse
from utils.argutils import parse_3d_args, parse_training_args, parse_general_args
import torch, os, time, datetime
from utils.utils import get_e2e_loaders
from e2e_handnet.e2e_handnet import E2EHandNet
from mano_train.evaluation.zimeval import EvalUtil
from progress.bar import Bar as Bar
from mano_train.exputils.monitoring import Monitor
from mano_train.evaluation.evalutils import AverageMeters
from datasets3d.queries import BaseQueries, TransQueries
from mano_train.visualize import displaymano
import pickle, numpy as np
import json
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
import matplotlib
from torchvision import transforms
from utils.hpe_eval import hpe_evaluate, hpe_plot
import logging

matplotlib.use('Agg') 

def reshape_output(outputs):

    for output in outputs:
        N = output['boxes'].shape[0]
        output['boxes'] = output['boxes'].reshape(N, 4)
        output['labels'] = output['labels'].reshape(N)
        output['scores'] = output['scores'].reshape(N)
        #output['sides'] = output['sides'].reshape(N)

    return outputs

def prepare_images(images, boxes):
    images = [ image.cpu() for image in images ]
    boxes = [ box.cpu() for box in boxes ]
    new_images = []

    for image, box in zip(images, boxes):
        box = box.numpy().squeeze()
        box = box.astype(np.int32)
        image = image[:, box[1]:box[3] + 1, box[0]:box[2] + 1,]
        t_image = transforms.ToPILImage()(image).resize((224, 224))
        t_image = transforms.ToTensor()(t_image)

        new_images.append(t_image)
    
    return torch.stack(new_images)

def evaluate(
    model,
    data_loader,
    args,
    start,
    end,
    fig, 
    faces_left,
    faces_right,
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

        all_boxes = []
        metric_path = os.path.join(args.output_dir, f'mano_test_metrics/')
        os.makedirs(metric_path, exist_ok=True)
        image_path_epoch = os.path.join(image_path, f'epoch_{epoch}')
        os.makedirs(image_path_epoch, exist_ok=True)

        with torch.inference_mode():
            for idx, (images, targets, sample, _) in enumerate(pbar):
                images = [ image.cuda() for image in images ]
                sample = {k: v.cuda() for k, v in sample.items()}
                model_time = time.time()
                results, detections, batch_idx, level_idx, boxes = model(images, targets, sample, is_3D=True)
                model_time = time.time() - model_time
                time_meters.add_loss_value("model_time", model_time)
                pbar.set_postfix({
                    'Model Time': time_meters.average_meters['model_time'].avg,
                    'Epoch': epoch,
                })

                if (idx % display_freq == 0):
                    for level, boxes_per_level in enumerate(boxes):
                        if len(boxes_per_level) == 0:
                            continue
                        sample[TransQueries.images] = prepare_images(images, boxes_per_level)
                        save_img_path = os.path.join(
                            image_path_epoch, f"level_{level}/img_{idx}.png"
                        )
                        os.makedirs(os.path.dirname(save_img_path), exist_ok=True)
                        displaymano.visualize_batch(
                                save_img_path,
                                fig=fig,
                                sample=sample,
                                results=results,
                                faces_right=faces_right,
                                faces_left=faces_left,
                                no_2d=True
                            )

                # outputs = detections

                # outputs = reshape_output(outputs)

                # pred_dict = {}
                # for i, output in enumerate(outputs):
                #     pred_dict["image_id"] = sample["dexycb_id"][i].item()
                #     scores = output["scores"].cpu().numpy()
                #     boxes = output["boxes"].cpu().numpy()
                #     labels = output["labels"].cpu().numpy()
                #     for j, bbox in enumerate(boxes):
                #         bbox[2:] -= bbox[:2]
                #         pred_dict["bbox"] = list(np.float64(bbox))
                #         pred_dict["score"] = np.float64(scores[j].item())
                #         pred_dict["category_id"] = labels[j].item()


                for i in range(len(results['joints'])):
                    joints3d = results['joints'][i]

                    joints3d = joints3d.cpu().detach().numpy()
                    j_text = ''
                    for j in joints3d:
                        j_text += str(list(j)).strip()[1:-1] + ','
                    j_text = j_text.replace(" ", "")[:-1]

                    os.makedirs(os.path.join(metric_path, f'level_{level_idx[i]}'), exist_ok=True)
                    
                    with open(metric_path + f'level_{level_idx[i]}/' + f's0_test_{epoch}.txt', 'a') as output:
                        print(str(sample['dexycb_id'][batch_idx[i]].cpu().numpy())[1:-1] + ',' + j_text, file=output)
        
        print(f"Average fps: {1./time_meters.average_meters['model_time'].avg}")

    for level in range(2, 3):
        hpe_evaluate('mano', args.output_dir, start, end, str(level))
        hpe_plot( args.output_dir, start, end, str(level))

    logger.setLevel(old_level)
            

def train_one_epoch_mano(
    model, 
    optimizer, 
    data_loader, 
    epoch, 
    scaler, 
    fig, 
    faces_left,
    faces_right,
    display_freq=5000
):
    avg_meters = AverageMeters()
    time_meters = AverageMeters()
    evaluator = EvalUtil(root_relative=True)

    save_img_folder = os.path.join(
           args.output_dir , "images", "train", "epoch_{}".format(epoch)
    )

    os.makedirs(save_img_folder, exist_ok=True)
    pbar = tqdm(data_loader)
    idxs = list(range(21))

    for idx, (images,targets, sample, _) in enumerate(pbar):
        end = time.time()
        images = list(img.cuda() for img in images)
        targets = [{k: v.cuda() for k, v in t.items()} for t in targets]
        sample = {k: v.cuda() for k, v in sample.items()}        
        time_meters.add_loss_value("data_time", time.time() - end)

        with torch.cuda.amp.autocast(enabled=scaler is not None):
            # model_losses, results, target = model(images, sample)
            model_losses, results, sample = model(images, targets, sample, is_3D=True)
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

        fig = plt.figure(figsize=(12, 4))
    
    with open("misc/mano/MANO_RIGHT.pkl", "rb") as p_f:
        mano_right_data = pickle.load(p_f, encoding="latin1")
        faces_right = mano_right_data["f"]
    with open("misc/mano/MANO_LEFT.pkl", "rb") as p_f:
        mano_left_data = pickle.load(p_f, encoding="latin1")
        faces_left = mano_left_data["f"]

    if args.test_only:
        
        evaluate(model, data_loader_test, args, 1, args.start_epoch - 1, fig, faces_left, faces_right)
        return


    print("Start training")
    start_time = time.time()

    hosting_folder = os.path.join(args.output_dir, "hosting")
    monitor = Monitor(args.output_dir, hosting_folder=hosting_folder)
    fig = plt.figure(figsize=(12, 12))

    for epoch in range(args.start_epoch, args.epochs+1):
        
        train_avg_meters, train_pck_infos = train_one_epoch_mano(
            model, 
            optimizer, 
            data_loader,
            epoch, 
            scaler,
            fig,
            faces_left,
            faces_right,
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
            save_name = os.path.join(output_dir, f'e2e_handnet_{epoch}.pth')
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
    parse_training_args(parser)
    parse_3d_args(parser)

    args = parser.parse_args()

    args.output_dir = args.save_dir + "/" + args.net + "_" + args.model_name + "_" + str(args.session)
    if 'res' in args.net :
        args.output_dir +=  "_fpn"
    print(f'\n---------> model output_dir = {args.output_dir}\n')

    os.makedirs(args.output_dir, exist_ok=True)

    main(args)