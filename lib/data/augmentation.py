import random
from lib.utils.tools import read_pkl
from lib.utils.utils_data import flip_data, crop_scale_3d


class Augmenter3D(object):
    def __init__(self, args):
        self.flip = args.flip
        if hasattr(args, "scale_range_pretrain"):
            self.scale_range_pretrain = args.scale_range_pretrain
        else:
            self.scale_range_pretrain = None

    def augment3D(self, motion_3d):
        if self.scale_range_pretrain:
            motion_3d = crop_scale_3d(motion_3d, self.scale_range_pretrain)
        if self.flip and random.random()>0.5:
            motion_3d = flip_data(motion_3d)
        return motion_3d
