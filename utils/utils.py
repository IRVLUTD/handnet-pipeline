from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roiFPNbatchLoader import *
from torchvision import transforms as T
import os, pickle

from datasets3d.detectdataset import DetectDataset
from datasets3d.e2edataset import E2EDataset
from fpn_utils.group_by_aspect_ratio import GroupedBatchSampler, create_aspect_ratio_groups

def collate_fn(batch):
    return tuple(zip(*batch))

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
        if args.aspect_ratio_group_factor >= 0:
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


def get_e2e_loaders(args, detect=False):
    cache_dir = 'data/e2e/cache_detect/dataloaders/' if detect else 'data/e2e/cache/dataloaders/'
    os.makedirs(cache_dir, exist_ok=True)

    print("Creating dataloaders")
    
    if os.path.exists(f'{cache_dir}train.pkl'):
        with open(f'{cache_dir}train.pkl', 'rb') as f:
            data_loader = pickle.load(f)
    else:
        dataset = DetectDataset(get_transform()) if detect else E2EDataset(get_transform())
        train_sampler = torch.utils.data.RandomSampler(dataset)
        if args.aspect_ratio_group_factor >= 0:
            group_ids = create_aspect_ratio_groups(dataset, k=args.aspect_ratio_group_factor)
            train_batch_sampler = GroupedBatchSampler(train_sampler, group_ids, args.batch_size)
        else:
            train_batch_sampler = torch.utils.data.BatchSampler(
                train_sampler, args.batch_size, drop_last=True)

        data_loader = torch.utils.data.DataLoader(
        dataset, batch_sampler=train_batch_sampler, num_workers=args.workers,
        collate_fn=collate_fn if detect else None)

        with open(f'{cache_dir}train.pkl', 'wb+') as f:
            pickle.dump(data_loader, f)
    
    if os.path.exists(f'{cache_dir}test.pkl'):
        with open(f'{cache_dir}test.pkl', 'rb') as f:
            data_loader_test = pickle.load(f)
    else:
        dataset_test = DetectDataset(get_transform(), train=False) if detect else E2EDataset(get_transform(), train=False)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

        data_loader_test = torch.utils.data.DataLoader(
            dataset_test, batch_size=1,
            sampler=test_sampler, num_workers=args.workers,
            collate_fn=collate_fn if detect else None)

        with open(f'{cache_dir}test.pkl', 'wb+') as f:
            pickle.dump(data_loader_test, f)

    print("Created dataloaders")
    return data_loader, data_loader_test