import os
import random

import torch
from torch.utils.data import Dataset

from lib.utils.tools import read_pkl
from lib.utils.utils_data import flip_data


class MotionDataset(Dataset):
    def __init__(self, args, subset_list, data_split):
        self.data_root = args.data_root
        self.subset_list = subset_list
        self.data_split = data_split
        self.file_list = []
        for subset in subset_list:
            data_path = os.path.join(self.data_root, subset, data_split)
            for filename in sorted(os.listdir(data_path)):
                self.file_list.append(os.path.join(data_path, filename))

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        raise NotImplementedError


class MotionDataset3D(MotionDataset):
    def __init__(self, args, subset_list, data_split):
        super().__init__(args, subset_list, data_split)
        self.flip = args.flip

    def __getitem__(self, index):
        motion_file = read_pkl(self.file_list[index])
        motion_2d = motion_file['data_input']
        motion_3d = motion_file['data_label']
        if motion_2d is None:
            raise ValueError('The Human3.6M sample does not contain 2D input.')

        if self.data_split == 'train':
            if self.flip and random.random() > 0.5:
                motion_2d = flip_data(motion_2d)
                motion_3d = flip_data(motion_3d)
        elif self.data_split != 'test':
            raise ValueError(f'Unsupported data split: {self.data_split}')

        return torch.as_tensor(motion_2d, dtype=torch.float32), torch.as_tensor(
            motion_3d, dtype=torch.float32
        )
