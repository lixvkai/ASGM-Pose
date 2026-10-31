import argparse
import hashlib
import json
import logging
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
import yaml

from lib.data.datareader_h36m import DataReaderH36M
from lib.data.dataset_motion_3d import MotionDataset3D
from lib.data.h36m_validation import build_train_sequence_split, evaluate_train_validation
from lib.utils.checkpoint import read_checkpoint, load_model_weights
from lib.utils.learning import AverageMeter, load_backbone
from lib.utils.tools import get_config


def _seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _new_run(args, seed):
    _seed(seed)
    model = nn.DataParallel(load_backbone(args)).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                  weight_decay=args.weight_decay)
    return model, optimizer


def _save(path, phase, epoch, model, optimizer, selected_epochs, best_val):
    torch.save({
        'phase': phase, 'epoch': epoch, 'model_pos': model.state_dict(),
        'optimizer': optimizer.state_dict(), 'lr': optimizer.param_groups[0]['lr'],
        'selected_epochs': selected_epochs, 'best_validation_loss': best_val,
        'random_state': random.getstate(), 'numpy_state': np.random.get_state(),
        'torch_state': torch.get_rng_state(),
        'cuda_state': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }, path)


def _restore(model, optimizer, checkpoint):
    load_model_weights(model, checkpoint)
    optimizer.load_state_dict(checkpoint['optimizer'])
    random.setstate(checkpoint['random_state'])
    np.random.set_state(checkpoint['numpy_state'])
    torch.set_rng_state(checkpoint['torch_state'])
    if checkpoint['cuda_state']:
        torch.cuda.set_rng_state_all(checkpoint['cuda_state'])


def _train(args, model, loader, optimizer, train_epoch):
    keys = ['3d_pos', '3d_scale', '2d_proj', 'lg', 'lv', 'total',
            '3d_velocity', 'angle', 'angle_velocity']
    losses = {key: AverageMeter() for key in keys}
    train_epoch(args, model, loader, losses, optimizer, has_3d=True, has_gt=True)
    for group in optimizer.param_groups:
        group['lr'] *= args.lr_decay
    return losses['total'].avg


def run_stages(args, seed, output, train_loader, val_loader, full_loader,
               train_epoch, validate, final_evaluate, logger, resume=None, training_mode='auto'):
    if training_mode not in ['auto', 'fixed']:
        raise ValueError('Unknown training mode')
    checkpoint = read_checkpoint(resume) if resume else None
    phase = 'fixed' if training_mode == 'fixed' else 'final'
    allowed_phases = ['fixed'] if training_mode == 'fixed' else ['validation', 'final']
    if checkpoint and checkpoint.get('phase') not in allowed_phases:
        raise ValueError('Resume checkpoint does not match the training mode')
    if training_mode == 'fixed':
        selected, best_val = args.epochs, None
        if checkpoint and int(checkpoint['selected_epochs']) != selected:
            raise ValueError('Resume duration does not match the requested epochs')
    elif checkpoint and checkpoint['phase'] == 'final':
        selected = int(checkpoint['selected_epochs'])
        best_val = float(checkpoint['best_validation_loss'])
    else:
        model, optimizer = _new_run(args, seed)
        start, selected, best_val = 0, 0, float('inf')
        if checkpoint:
            _restore(model, optimizer, checkpoint)
            start = int(checkpoint['epoch'])
            selected = int(checkpoint['selected_epochs'])
            best_val = float(checkpoint['best_validation_loss'])
        if start > args.epochs:
            raise ValueError('Resume epoch exceeds the validation budget')
        for epoch in range(start + 1, args.epochs + 1):
            loss = _train(args, model, train_loader, optimizer, train_epoch)
            value = float(validate(args, model, val_loader, logger))
            if not math.isfinite(value):
                raise RuntimeError('Validation loss is not finite')
            if value < best_val:
                selected, best_val = epoch, value
                _save(output / 'best_validation_epoch.bin', 'validation', epoch,
                      model, optimizer, selected, best_val)
            _save(output / 'latest_validation_epoch.bin', 'validation', epoch,
                  model, optimizer, selected, best_val)
            if epoch % args.checkpoint_frequency == 0:
                _save(output / f'validation_epoch_{epoch}.bin', 'validation', epoch,
                      model, optimizer, selected, best_val)
            logger.info('validation epoch=%d train_loss=%.8f val_mpjpe=%.8f',
                        epoch, loss, value)
        del model, optimizer
        torch.cuda.empty_cache()
    if not 1 <= selected <= args.epochs:
        raise ValueError('Training duration is outside the epoch budget')
    model, optimizer = _new_run(args, seed)
    start = 0
    if checkpoint and checkpoint['phase'] == phase:
        _restore(model, optimizer, checkpoint)
        start = int(checkpoint['epoch'])
    if start > selected:
        raise ValueError('Resume epoch exceeds the selected final duration')
    for epoch in range(start + 1, selected + 1):
        loss = _train(args, model, full_loader, optimizer, train_epoch)
        _save(output / 'latest_final_epoch.bin', phase, epoch,
              model, optimizer, selected, best_val)
        if epoch % args.checkpoint_frequency == 0:
            _save(output / f'final_epoch_{epoch}.bin', phase, epoch,
                  model, optimizer, selected, best_val)
        logger.info('final epoch=%d/%d train_loss=%.8f', epoch, selected, loss)
    _save(output / 'best_epoch.bin', phase, selected,
          model, optimizer, selected, best_val)
    p1, p2 = final_evaluate(model)
    result = {'selected_epochs': selected, 'best_validation_mpjpe_normalized': best_val,
              'test_mpjpe_mm': float(p1), 'test_p_mpjpe_mm': float(p2)}
    return result


def main(train_epoch, evaluate):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/h36m/ASGM_Pose_B.yaml')
    parser.add_argument('-c', '--checkpoint', default='checkpoints/h36m_B')
    parser.add_argument('--data-root')
    parser.add_argument('--metadata-root')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--split-seed', type=int, default=0)
    parser.add_argument('--heldout-sequences', type=int, default=2)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--resume')
    parser.add_argument('--training-mode', choices=['auto', 'fixed'], default='auto',
                        help='auto: validation then full retraining; fixed: full training only')
    parser.add_argument('--epochs', type=int,
                        help='Override YAML epochs: validation budget in auto, duration in fixed')
    opts = parser.parse_args()
    if opts.workers < 0 or opts.heldout_sequences < 1:
        parser.error('workers must be nonnegative and heldout-sequences must be positive')
    args = get_config(opts.config)
    config_epochs = args.epochs
    if opts.epochs is not None:
        args.epochs = opts.epochs
    if opts.data_root:
        args.data_root = opts.data_root
    if opts.metadata_root:
        args.dt_root = opts.metadata_root
    if args.epochs < 1 or args.checkpoint_frequency < 1:
        parser.error('epochs and checkpoint_frequency must be positive')
    unsupported = ['train_2d', 'finetune', 'partial_train', 'synthetic', 'gt_2d', 'noise']
    if any(getattr(args, key, False) for key in unsupported):
        raise ValueError('This pipeline expects the supplied Human3.6M SH configurations')
    args.mask = args.mask_ratio > 0 and args.mask_T_ratio > 0
    if args.mask:
        raise ValueError('Stochastic masking is not supported by this validation protocol')
    if not torch.cuda.is_available():
        raise RuntimeError('Training requires CUDA')
    output = Path(opts.checkpoint).resolve()
    settings = {
        'config_sha256': hashlib.sha256(Path(opts.config).read_bytes()).hexdigest(),
        'training_mode': opts.training_mode, 'epochs': args.epochs,
        'seed': opts.seed, 'split_seed': opts.split_seed,
        'heldout_sequences': opts.heldout_sequences,
        'data_root': str(Path(args.data_root).resolve()),
        'metadata_root': str(Path(args.dt_root).resolve()),
    }
    resume = Path(opts.resume).resolve() if opts.resume else None
    if resume:
        if resume.parent != output or not resume.is_file():
            raise ValueError('Resume checkpoint must exist in the specified run directory')
        protocol = json.loads((output / 'protocol.json').read_text())
        saved_settings = dict(protocol['settings'])
        saved_settings.setdefault('training_mode', 'auto')
        saved_settings.setdefault('epochs', config_epochs)
        if saved_settings != settings:
            raise ValueError('Resume settings do not match the saved protocol')
    else:
        output.mkdir(parents=True, exist_ok=False)
        with (output / 'config.yaml').open('w') as f:
            yaml.safe_dump(dict(args), f, sort_keys=False)
    logger = logging.getLogger('ASGM-Pose-training')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter('%(asctime)s %(message)s')
    for handler in [logging.StreamHandler(), logging.FileHandler(output / 'run.log')]:
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    reader = DataReaderH36M(
        n_frames=args.clip_len, sample_stride=args.sample_stride,
        data_stride_train=args.data_stride, data_stride_test=args.clip_len,
        dt_root=args.dt_root, dt_file=args.dt_file,
    )
    full_dataset = MotionDataset3D(args, args.subset_list, 'train')
    train_indices = list(range(len(full_dataset)))
    val_indices = []
    val_dataset_raw = None
    if opts.training_mode == 'auto':
        val_dataset_raw = MotionDataset3D(args, args.subset_list, 'train')
        val_dataset_raw.flip = False
        if any(Path(name).name != f'{i:08d}.pkl' for i, name in enumerate(full_dataset.file_list)):
            raise ValueError('Training clips must follow the original contiguous index order')
        train_indices, val_indices, _ = build_train_sequence_split(
            reader, len(full_dataset), opts.split_seed, opts.heldout_sequences,
        )
        if set(train_indices) & set(val_indices) or len(train_indices) + len(val_indices) != len(full_dataset):
            raise RuntimeError('Training and validation clips must be disjoint and exhaustive')
    if not resume:
        (output / 'protocol.json').write_text(json.dumps({
            'settings': settings, 'training_subjects': [1, 5, 6, 7, 8],
            'test_subjects': [9, 11], 'training_clips': len(train_indices),
            'validation_clips': len(val_indices), 'final_training_clips': len(full_dataset),
            'selection_metric': ('validation MPJPE in normalized training coordinates'
                                 if opts.training_mode == 'auto' else None),
            'final_training': 'fresh model and optimizer on all official training clips',
            'duration_rule': 'validation-selected' if opts.training_mode == 'auto' else 'user-specified',
        }, indent=2))
    loader_settings = {'num_workers': opts.workers, 'pin_memory': True}
    if opts.workers:
        loader_settings.update(prefetch_factor=4, persistent_workers=True)
    train_loader = val_loader = None
    if opts.training_mode == 'auto':
        train_loader = DataLoader(Subset(full_dataset, train_indices),
                                  batch_size=args.batch_size, shuffle=True, **loader_settings)
        val_loader = DataLoader(Subset(val_dataset_raw, val_indices),
                                batch_size=args.batch_size, shuffle=False, **loader_settings)
    full_loader = DataLoader(full_dataset, batch_size=args.batch_size,
                             shuffle=True, **loader_settings)

    def final_evaluate(model):
        test_dataset = MotionDataset3D(args, args.subset_list, 'test')
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                                 shuffle=False, **loader_settings)
        evaluate.__globals__['log'] = logger
        p1, p2, _ = evaluate(args, model, test_loader, reader)
        return p1, p2

    run_stages(args, opts.seed, output, train_loader, val_loader, full_loader,
               train_epoch, evaluate_train_validation, final_evaluate, logger, resume, opts.training_mode)
