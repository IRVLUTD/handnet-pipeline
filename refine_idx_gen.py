from dex_ycb_toolkit.factory import get_dataset

import pickle
import numpy as np
import os
from tqdm import tqdm

for split in ['s0_train', 's0_val', 's0_test']:
    data = get_dataset(split)

    refine_idx = []

    for idx, sample in enumerate(tqdm(data)):
        label = np.load(sample['label_file'])

        h, w = 480, 640
        jnt_uv = label['joint_2d'].squeeze() + 1
        ustart, uend, vstart, vend = (0, h, 0, w)

        x_out = (jnt_uv[:, 0] < vstart).sum() + (jnt_uv[:, 0] > vend).sum()
        y_out = (jnt_uv[:, 1] < ustart).sum() + (jnt_uv[:, 1] > uend).sum()
        root_out = (jnt_uv[0, 0] < vstart).sum() + (jnt_uv[0, 0] > vend).sum() + (jnt_uv[0, 1] < ustart).sum() + (jnt_uv[0, 1] > uend).sum()
        if not jnt_uv.any() or x_out > 2 or y_out > 2 or root_out > 0:
            pass
        else:
            refine_idx.append(idx)

    split_name = split[3:]
    os.makedirs('data/e2e/cache/', exist_ok=True)
    pickle.dump(refine_idx, open(f'data/e2e/cache/refined_{split_name}_idx.pkl', 'wb+'))
