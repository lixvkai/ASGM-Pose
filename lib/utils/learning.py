class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def load_backbone(args):
    from lib.model.ASGM_Pose import ASGM_Pose
    if args.backbone != 'ASGM_Pose':
        raise ValueError('Expected backbone: ASGM_Pose')
    return ASGM_Pose(num_frame=args.maxlen, num_joints=args.num_joints,
        in_chans=2 if args.no_conf else 3, embed_dim_ratio=args.dim_feat,
        depth=args.depth, mlp_ratio=args.mlp_ratio, token_num=args.token_num,
        layer_index=args.layer_index, drop_rate=getattr(args, 'dropout', 0),
        drop_path_rate=getattr(args, 'drop_path_rate', 0.2))
