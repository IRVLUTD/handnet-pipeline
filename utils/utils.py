import matplotlib
from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roiFPNbatchLoader import *
from torchvision import transforms as T
import os, pickle

from datasets3d.detectdataset import DetectDataset
from datasets3d.e2edataset import E2EDataset
from datasets3d.a2jdataset import A2JDataset
from fpn_utils.group_by_aspect_ratio import GroupedBatchSampler, create_aspect_ratio_groups
import matplotlib.pyplot as plt
import cv2

def collate_fn(batch):
    return tuple(zip(*batch))

def e2e_collate_fn(batch):
    batch = tuple(zip(*batch))
    return tuple([*batch[:2], torch.utils.data.dataloader.default_collate(batch[2]), torch.utils.data.dataloader.default_collate(batch[3]), torch.utils.data.dataloader.default_collate(batch[4])])

def get_transform():
    transforms = []
    transforms.append(T.ToTensor())
    return T.Compose(transforms)

def get_dataset(cache_dir, train):
    
    cache_dir = os.path.dirname(cache_dir[:-1]) + '/'
    os.makedirs(cache_dir, exist_ok=True)

    if train:
        split = 'trainval'
    else:
        split = 'test'
    imdb_name = f'voc_2007_{split}'

    if os.path.exists(f'{cache_dir}roidb_{split}.pkl'):
        with open(f'{cache_dir}roidb_{split}.pkl', 'rb') as f:
            imdb, roidb = pickle.load(f)
    else:
        imdb, roidb, _, _ = combined_roidb(imdb_name, train)
        with open(f'{cache_dir}roidb_{split}.pkl', 'wb+') as f:
            pickle.dump(tuple([imdb, roidb]), f)
    
    return roiFPNbatchLoader(roidb, get_transform()), imdb

def get_loaders_100doh(args):
    ## Cache detection batch loaders due to aspect ratio grouping
    cache_dir = 'data/100doh/cache/dataloaders/'
    os.makedirs(cache_dir, exist_ok=True)

    print("Creating dataloaders")
    
    if os.path.exists(f'{cache_dir}trainval.pkl'):
        with open(f'{cache_dir}trainval.pkl', 'rb') as f:
            imdb, data_loader = pickle.load(f)
            num_classes = imdb.num_classes
    else:
        dataset, imdb = get_dataset(cache_dir, train=True)
        num_classes = imdb.num_classes
        train_sampler = torch.utils.data.RandomSampler(dataset)
        if args.aspect_ratio_group_factor > 0:
            group_ids = create_aspect_ratio_groups(dataset, k=args.aspect_ratio_group_factor)
            train_batch_sampler = GroupedBatchSampler(train_sampler, group_ids, args.batch_size)
        else:
            train_batch_sampler = torch.utils.data.BatchSampler(
                train_sampler, args.batch_size, drop_last=True)

        data_loader = torch.utils.data.DataLoader(
        dataset, batch_sampler=train_batch_sampler, num_workers=args.workers,
        collate_fn=collate_fn)

        with open(f'{cache_dir}trainval.pkl', 'wb+') as f:
            pickle.dump(tuple([imdb,data_loader]), f)
    
    if os.path.exists(f'{cache_dir}test.pkl'):
        with open(f'{cache_dir}test.pkl', 'rb') as f:
            imdb_test, data_loader_test = pickle.load(f)
            num_classes = imdb_test.num_classes
    else:
        dataset_test, imdb_test = get_dataset(cache_dir, train=False)
        num_classes = imdb_test.num_classes
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

        data_loader_test = torch.utils.data.DataLoader(
            dataset_test, batch_size=1,
            sampler=test_sampler, num_workers=args.workers,
            collate_fn=collate_fn)

        with open(f'{cache_dir}test.pkl', 'wb+') as f:
            pickle.dump(tuple([imdb_test,data_loader_test]), f)

    print("Created dataloaders")

    return data_loader, data_loader_test, imdb, imdb_test, num_classes


def get_e2e_loaders(args, detect=False, a2j=False):
    if detect:
        cache_dir = 'data/detect/cache/dataloaders/'
    elif a2j:
        cache_dir = 'data/a2j/cache/dataloaders/'
    else:
        cache_dir = 'data/e2e/cache/dataloaders/'
    os.makedirs(cache_dir, exist_ok=True)

    print("Creating dataloaders")
    
    if os.path.exists(f'{cache_dir}train.pkl'):
        with open(f'{cache_dir}train.pkl', 'rb') as f:
            data_loader = pickle.load(f)
    else:
        if detect:
            dataset = DetectDataset(get_transform())
            collate = collate_fn
        elif a2j:
            dataset = A2JDataset()
            collate = torch.utils.data.dataloader.default_collate
        else:
            dataset = E2EDataset(get_transform())
            collate = e2e_collate_fn
        train_sampler = torch.utils.data.RandomSampler(dataset)
        if args.aspect_ratio_group_factor > 0:
            group_ids = create_aspect_ratio_groups(dataset, k=args.aspect_ratio_group_factor)
            train_batch_sampler = GroupedBatchSampler(train_sampler, group_ids, args.batch_size)
        else:
            train_batch_sampler = torch.utils.data.BatchSampler(
                train_sampler, args.batch_size, drop_last=True)

        data_loader = torch.utils.data.DataLoader(
        dataset, batch_sampler=train_batch_sampler, num_workers=args.workers,
        collate_fn=collate)

        with open(f'{cache_dir}train.pkl', 'wb+') as f:
            pickle.dump(data_loader, f)
    
    if os.path.exists(f'{cache_dir}test.pkl'):
        with open(f'{cache_dir}test.pkl', 'rb') as f:
            data_loader_test = pickle.load(f)
    else:
        if detect:
            dataset = DetectDataset(get_transform(), train=False)
            collate = collate_fn
        elif a2j:
            dataset = A2JDataset(train=False)
            collate = torch.utils.data.dataloader.default_collate
        else:
            dataset = E2EDataset(get_transform(), train=False)
            collate = e2e_collate_fn

        test_sampler = torch.utils.data.SequentialSampler(dataset)

        data_loader_test = torch.utils.data.DataLoader(
            dataset, batch_size=1,
            sampler=test_sampler, num_workers=args.workers,
            collate_fn=collate)

        with open(f'{cache_dir}test.pkl', 'wb+') as f:
            pickle.dump(data_loader_test, f)

    print("Created dataloaders")
    return data_loader, data_loader_test

def vis_minibatch(images, depths, jt_gt, vis_tool, dexycb_id, path='vis.jpg', jt_pred=None):
    #matplotlib.use("qt5agg")

    m = len(images)
    n = 4

    jt_gt = jt_gt.reshape(-1, 21, 3)
    jt_pred = jt_pred.reshape(-1, 21, 3)
    
    figs = [ plt.subplots(4 if len(images) > 4 else len(images), 3, figsize=(6,6)) for i in range(len(images)//4 if len(images) > 4 else 1) ]
    for i, im in enumerate(images):

        # image
        # im = images[i, :3, :, :].copy()
        # im = im.transpose((1, 2, 0)) * 255.0
        # im += cfg.PIXEL_MEANS
        # im = im[:, :, (2, 1, 0)]
        # im = np.clip(im, 0, 255)
        # im = im.astype(np.uint8)

        '''
        if out_label_refined is not None:
            mask = out_label_refined_blob[i, :, :]
            visualize_segmentation(im, mask)
        #'''

        # show image
        (ax1, ax2, ax3) = figs[i // 4][1][i%4] if len(images) > 1 else figs[i // 4][1]
        im = im.squeeze()
        im = np.clip(im, 0, 255)
        im = im.astype(np.uint8)
        # convert im to BGR
        im = im[:, :, (2, 1, 0)]
        ax1.imshow(im)
        ax1.set_title(f'image {dexycb_id[i]}')
        ax1.axis('off')
        
        if depths is not None:
            depth = depths[i]
            depth = depth.squeeze()
            #depth = vis_tool.orig_depth(depth)
            #depth = np.clip(depth, 0, 255)
            #depth = depth.astype(np.uint8)
            ax2.imshow(depth)
            ax2.set_title('depth')
            ax2.axis('off')

        # show label
        if jt_gt is not None:
            label = jt_gt[i, :, :]
            c_label = vis_tool.plot(im, path=None, jt_uvd_gt=label, jt_uvd_pred=jt_pred[i, :, :] if jt_pred is not None else None, return_image=True)
            c_label = np.clip(c_label, 0, 255)
            c_label = c_label.astype(np.uint8)
            ax3.imshow(c_label)
            ax3.set_title('joints_on_color')
            ax3.axis('off')

    image_plots = []
    for fig, _ in figs:
        fig.tight_layout(pad=0.1)
        fig.canvas.draw()
        image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        # bgr to rgb
        image_from_plot = image_from_plot[:, :, (2, 1, 0)]
        image_plots.append(image_from_plot)
    
    img = cv2.hconcat(image_plots)
    cv2.imwrite(path, img)
    plt.close('all')
    #plt.savefig(path, dpi=300, bbox_inches='tight')
    #plt.show()