import argparse
from PIL import Image, ImageDraw, ImageFont

from matplotlib import pyplot as plt
from datasets3d.queries import BaseQueries, TransQueries

import cv2

from mano_train.exputils import argutils
from progress.bar import Bar as Bar


from model.utils.net_utils import vis_detections_filtered_objects
from multiprocessing import Process
from mano_train.demo.preprocess import prepare_input, preprocess_frame
import numpy as np
import os, pickle
import time
from copy import deepcopy
from mano_train.visualize import displaymano
from mano_train.modelutils import modelio
from e2e_handnet.e2e_handnet import E2EHandNet
import torch
import math
from mano_train.evaluation.evalutils import AverageMeters
import torchvision.transforms as transforms

def gentext(meta):
    hand_idx, side = meta
    
    font = ImageFont.truetype('lib/model/utils/times_b.ttf', size=12)
    text = f"{'Left' if side else 'Right'} #{hand_idx}"
    
    curr = deepcopy(white_text)
    draw = ImageDraw.Draw(curr)
    w1, h1 = draw.textsize(text, font=font)
    draw.text(((int(frame_h/2)-w1)/2,(20-h1)/2), text, fill="black", font=font)
    curr = np.array(curr)
    curr = cv2.cvtColor(curr, cv2.COLOR_BGR2RGBA)

    return curr


def createframe(mode=0,frame=[], meshes=[], meta=[]):
    if mode==0:
        return cv2.hconcat([frame, white0])
    elif mode==1:
        vert_meshes = [cv2.cvtColor(frame, cv2.COLOR_RGB2RGBA)]
        for i in range(0,len(meshes),2):
            top = meshes[i]
            bot = meshes[i+1]
            vert_meshes.append(cv2.vconcat([ white1[int(meta[i][1] == True)][meta[i][0]-1], top, bot, white1[int(meta[i+1][1]==True)][meta[i+1][0]-1]]))
        if len(meshes) < mhands: vert_meshes.append(rest[int(len(meshes)/2)-1])
        return cv2.hconcat(vert_meshes)
    elif mode==2:
        vert_meshes = [cv2.cvtColor(frame, cv2.COLOR_RGB2RGBA)]
        last = len(meshes) - 1
        for i in range(0,last,2):
            top = meshes[i]
            bot = meshes[i+1]
            vert_meshes.append(cv2.vconcat([ white1[int(meta[i][1]==True)][meta[i][0]-1], top, bot, white1[int(meta[i+1][1]==True)][meta[i+1][0]-1]]))
        if last: vert_meshes.append(cv2.vconcat([top, odd_white]))
        if len(meshes) < mhands: vert_meshes.append(rest[int((len(meshes)+1)/2) - 1])
        return cv2.hconcat(vert_meshes)

def create_mesh_frame(mesh):
       

if __name__ == "__main__":
    global frame_w, frame_h, w, h, white0, white1, rest, white_text, odd_white, mhands, display_mesh
    white1 = [[],[]]
    rest = []


    # init frames
    frames, det_frames, dets_arr, mesh_frames = [], [], [], []

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r", "--resume",
        type=str,
        help="Path to checkpoint",
        required=True
    )
    parser.add_argument("--video_path", help="Path to video", required=True)  
    args = parser.parse_args()
    argutils.print_args(args)

    display_mesh = args.display_mesh

    # Init CV2 Video Writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    current_directory = os.getcwd()
    output_directory = os.path.join(current_directory, 'output/')
    
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)
    iternum = 1
    while os.path.exists(output_directory + str(iternum) + '.mp4'):
        iternum+=1

    # Load model options
    checkpoint = os.path.dirname(args.resume)
    with open(os.path.join(checkpoint, "opt.pkl"), "rb") as opt_f:
        opts = pickle.load(opt_f)
    
    # Load faces of hand
    with open("misc/mano/MANO_RIGHT.pkl", "rb") as p_f:
        mano_right_data = pickle.load(p_f, encoding="latin1")
        faces = mano_right_data["f"]

    print(" ------------------- Load E2E HandNet ------------------- \n")
    model = E2EHandNet(args)
    model.cuda()

    # Load checkpoint
    checkpoint = torch.load(args.resume, map_location="cpu")
    model.load_state_dict(checkpoint["model"])

    model.eval()
    
    cap = cv2.VideoCapture(args.video_path)
    print(" ------------------- Reading Video ------------------- \n")
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    len_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


    if cap is None:
        raise RuntimeError("OpenCV could not read video")
    det_frames = []
    
    start = time.time()
    time_meters = AverageMeters()
    bar = Bar("Processing", max=len_frames)

    transform = transforms.ToTensor()
    mhands = 0
    writer = cv2.VideoWriter(output_directory + str(iternum) + '.mp4', fourcc, 20, (w, h))
    w = frame_w + (int(math.floor(mhands/2)*math.floor(frame_h/2)) if mhands%2==0 else int(math.floor(mhands/2+1)*math.floor(frame_h/2)))
    h = frame_h

    white0 = np.zeros([h, w-frame_w,3], 255, dtype=np.uint8)
    white_text = np.full([20, int(frame_h/2),3], 255, dtype=np.uint8)
    white_text = white_text[:,:,::-1]
    white_text = Image.fromarray(white_text).convert("RGB")
    for i in range(1, mhands+1):
        white1[0].append(gentext((i, False)))
        white1[1].append(gentext((i, True)))

        if i%2==0:
            rest.append( white0[:,0: (w-frame_w - int(frame_h/2) * int(i/2))] ) 

    odd_white =  np.full( [int(frame_h/2)+20, int(frame_h/2),3], 255, dtype=np.uint8)

    frames = []
    frames_data = []
    frames_n_hands = []

    for framenum in range(len_frames):
        # frame = cv2.resize(frame, (640, 480))
        frame_time = time.time()
        ret, frame = cap.read()
        frames.append(frame)

        inp_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        inp_frame = transform(inp_frame).cuda()
    
        model_time = time.time()
        with torch.inference_mode():
            results, detections, batch_idx, level_idx, boxes = model([frame])
        model_time = time.time() - model_time

        time_meters.add_loss_value("model_time", model_time)

        obj_ind = [torch.nonzero( (t['scores'] > 0.1) & (t['labels'] != 22) ).squeeze() for t in detections]
        hand_ind = [torch.nonzero( (t['scores'] > 0.1) & (t['labels'] == 22) ).squeeze() for t in detections]

        mhands = max(mhands, hand_ind[0].shape[0])
        frames_n_hands.append(hand_ind[0].shape[0])

        obj_final = [
            torch.cat((
                t['boxes'][i],
                t['scores'][i],
                t['contacts'][i],
                t['dxdymags'][i],
                t['sides'][i],
                torch.ones_like(t['sides'][i]) # filler for nc_prob
            ), -1).reshape(-1, 11).cpu().numpy()
            for t, i in zip(detections, obj_ind)
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
            for t, i in zip(detections, hand_ind)
        ]

        det_frame = vis_detections_filtered_objects(frame, obj_final, hand_final, 0.5)

        hand_level_idxs = [
            torch.where ( (t['labels' == 2]) & (t['feature_idx'] == feat_idx) ) for t in detections
            for feat_idx in range(len(3))
        ]

        mesh_frames = []
        meta = []

        for mesh_idx, mesh in enumerate(results[TransQueries.verts3d]):
            mesh = mesh.cpu().numpy()
            lev_idx = level_idx[mesh_idx]
            hand_idx = hand_level_idxs[lev_idx][mesh_idx]
            meta.append( (hand_idx, detections[0]['sides'][hand_idx]) )
            mesh_frame = create_mesh_frame(mesh)
            mesh_frames.append(mesh_frame)

        frames_data.append( (det_frame, mesh_frames, meta) )
        
        frame_time = time.time() - frame_time
        time_meters.add_loss_value("frame_time", frame_time)

        bar.suffix = f"({framenum + 1}/{len_frames}) Frame: {time_meters.average_meters['frame_time'].avg}s | Model Time: {time_meters.average_meters['model_time'].avg}s | Total: {bar.elapsed_td} | ETA: {bar.eta_td}"
        bar.next()

    cap.release()

    for framenum in range(len_frames):
        if frames_n_hands[framenum] != 0:
            det_frame, mesh_frames, meta = frames_data[framenum]
            frame = createframe(mode=1 if len(mesh_frames)%2==0 else 2, frame=det_frame, meshes=mesh_frames, meta=meta)
        else:
            frame = createframe(mode=0, frame=frame)
    writer.write(frame)
    
    writer.release()
    cv2.destroyAllWindows()


    # print(" ------------------- Start Mesh Reconstruction ------------------- \n")
    # for i in range(len(frames)):
    #     hand_dets = dets_arr[i]
    #     frame = frames[i]
    #     det_frame = det_frames[i]
    #     if type(hand_dets) is int: 
    #         mesh_frames.append(createframe(mode=0, frame=frame))
    #         continue
    #     hand_dets = [(hand_idx + 1, hand_dets[i, :]) for hand_idx, i in enumerate(range(np.minimum(10, hand_dets.shape[0]))) ]
    #     hands = [(hand_idx, crop(frame, det, 1.2), det[-1]) for hand_idx, det in hand_dets]
    #     # [
    #     #     cv2.imshow(f"Hand #{hand_idx}", frame)
    #     #     for hand_idx, frame, side in hands
    #     # ]
    #     hands = [(hand_idx, cv2.resize(preprocess_frame(frame), (256, 256)), not bool(side)) for hand_idx, frame, side in hands]
    #     hands_input = [(hand_idx, prepare_input(frame, flip_left_right=not side,), side) for hand_idx, frame, side in hands]


    #     samples = [
    #         (forward_pass_3d(hand, left=side), hand_idx, side)
    #         for hand_idx, hand, side in hands_input
    #     ]

    #     meta = [i[1:3] for i in samples]
    #     start = time.time()
    #     results= np.copy(ray.get([HandNets[i%mhands].forward.remote(samples[i][0], no_loss=True) for i in range(len(samples))]))
    #     mesh_end = time.time()
    #     meshes = np.copy(ray.get([plot.remote(hands[i], results[i][1]["verts"].cpu().detach().numpy()[0], figs[i]) for i in range(len(results))]))
    #     frame_end = time.time()
    #     mesh_frames.append((meshes, 1 if len(meshes)%2==0 else 2, det_frame, meta))
    #     del results
    #     del meshes
        
    #     print(f"\n\nFrame #{i+1} complete\nMesh Time: {(mesh_end - start)}\nPlot Time: {(frame_end - mesh_end)}")