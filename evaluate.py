import argparse
import logging
import torch
from torch.utils.data import DataLoader
from lib.utils.tools import get_config
from lib.utils.learning import load_backbone
from lib.utils.checkpoint import read_checkpoint, load_model_weights


def main():
    parser = argparse.ArgumentParser(description='ASGM-Pose dataset inference')
    parser.add_argument('--dataset', choices=['h36m'], default='h36m')
    parser.add_argument('--config', default='configs/h36m/ASGM_Pose_B.yaml')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data-root', help='Override preprocessed Human3.6M clip root')
    parser.add_argument('--metadata-root', help='Override Human3.6M metadata root')
    parser.add_argument('--workers', type=int, default=4)
    opts = parser.parse_args()
    if opts.workers < 0:
        parser.error('--workers must be nonnegative')
    args = get_config(opts.config)
    if opts.data_root:
        args.data_root = opts.data_root
    if opts.metadata_root:
        args.dt_root = opts.metadata_root
    if not torch.cuda.is_available():
        raise RuntimeError('The selective-scan implementation requires CUDA')
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    log = logging.getLogger('ASGM-Pose')
    model = load_backbone(args).cuda()
    load_model_weights(model, read_checkpoint(opts.checkpoint))
    model.eval()
    log.info('Checkpoint: %s', opts.checkpoint)
    log.info('Parameters: %d', sum(p.numel() for p in model.parameters()))
    import train_h36m as evaluator
    from lib.data.dataset_motion_3d import MotionDataset3D
    from lib.data.datareader_h36m import DataReaderH36M
    dataset = MotionDataset3D(args, args.subset_list, 'test')
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=opts.workers, pin_memory=True)
    reader = DataReaderH36M(n_frames=args.clip_len,
        sample_stride=args.sample_stride, data_stride_train=args.data_stride,
        data_stride_test=args.clip_len, dt_root=args.dt_root, dt_file=args.dt_file)
    evaluator.log = log
    p1, p2, _ = evaluator.evaluate(args, model, loader, reader)
    log.info('RESULT MPJPE=%.9f P-MPJPE=%.9f mm', p1, p2)


if __name__ == '__main__':
    main()
