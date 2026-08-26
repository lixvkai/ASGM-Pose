import random

from lib.utils.utils_data import crop_scale_3d, flip_data


class Augmenter3D:
    def __init__(self, args):
        self.flip = args.flip
        self.scale_range_pretrain = getattr(args, "scale_range_pretrain", None)

    def augment3D(self, motion_3d):
        if self.scale_range_pretrain:
            motion_3d = crop_scale_3d(motion_3d, self.scale_range_pretrain)
        if self.flip and random.random() > 0.5:
            motion_3d = flip_data(motion_3d)
        return motion_3d
