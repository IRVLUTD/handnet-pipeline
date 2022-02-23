from roi_data_layer.roidb import combined_roidb
import pprint
import torch
from torchvision import transforms as T
from model.utils.config import cfg, cfg_from_file
from roi_data_layer.roiFPNbatchLoader import *
import os, pickle

def get_transform():
    transforms = []
    transforms.append(T.ToTensor())
    return T.Compose(transforms)



cfg_from_file('cfgs/res101.yml')

print('Using config:')
pprint.pprint(cfg)

cfg.TRAIN.USE_FLIPPED = False
cfg.USE_GPU_NMS = True
imdb_name = 'voc_2007_trainval'

cache_dir = 'data/cache/'
os.makedirs(cache_dir, exist_ok=True)

if os.path.exists(f'{cache_dir}roidb_trainval.pkl'):
    with open(f'{cache_dir}roidb_trainval.pkl', 'rb') as f:
        roidb = pickle.load(f)
else:
    imdb, roidb, ratio_list, ratio_index = combined_roidb(imdb_name)
    with open(f'{cache_dir}roidb_trainval.pkl', 'wb+') as f:
        pickle.dump(roidb, f)

train_set = roiFPNbatchLoader(roidb, get_transform())
loader = torch.utils.data.DataLoader(train_set, batch_size=1, num_workers=1)
numpixels = 0
total_sum = 0
for i, batch in enumerate(loader):
    total_sum += torch.sum(torch.sum(batch[0].squeeze(), dim=2), dim=1)
    numpixels += torch.full((3,), batch[0].squeeze().numel() / 3)
    print(i)
mean = total_sum / numpixels
sum_of_squared_error = 0
print("mean: ", mean)
for i, batch in enumerate(loader):
    sum_of_squared_error += torch.sum(torch.sum((batch[0].squeeze() - mean[:, None, None]).pow(2), dim=2), dim=1)
    print(i)
std = torch.sqrt(sum_of_squared_error / numpixels)

print(mean)
print(std)

with open(f'{cache_dir}roidb_trainval_mean_std.pkl', 'wb+') as f:
    pickle.dump({'mean':mean, 'std':std}, f)
