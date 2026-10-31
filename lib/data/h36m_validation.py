import random
import numpy as np
import torch
from lib.utils.utils_data import flip_data


def build_train_sequence_split(reader, train_dataset_size, split_seed, heldout_per_subject):
    train_split_ids, _ = reader.get_split_id()
    if len(train_split_ids) != train_dataset_size:
        raise RuntimeError(f'Train index mismatch: reader has {len(train_split_ids)} clips, whereas MotionDataset3D has {train_dataset_size}.')
    clip_sources = np.asarray(reader.dt_dataset['train']['source'])[train_split_ids]
    groups = {}
    for clip_idx, clip_source in enumerate(clip_sources):
        source = str(clip_source[0])
        sequence = source.rsplit('_cam_', 1)[0]
        subject = int(sequence.split('_')[1])
        groups.setdefault(subject, {}).setdefault(sequence, []).append(clip_idx)
    if set(groups) != {1, 5, 6, 7, 8}:
        raise ValueError('Validation must be drawn only from the official training subjects')
    rng = random.Random(split_seed)
    validation_indices = []
    selected = {}
    for subject in sorted(groups):
        sequences = sorted(groups[subject])
        if len(sequences) <= heldout_per_subject:
            raise RuntimeError(f'S{subject} has only {len(sequences)} complete sequences; cannot reserve {heldout_per_subject}.')
        selected_sequences = sorted(rng.sample(sequences, heldout_per_subject))
        selected[f'S{subject}'] = [{'sequence': sequence, 'clips': len(groups[subject][sequence])} for sequence in selected_sequences]
        for sequence in selected_sequences:
            validation_indices.extend(groups[subject][sequence])
    validation_indices = sorted(validation_indices)
    validation_set = set(validation_indices)
    train_indices = [i for i in range(train_dataset_size) if i not in validation_set]
    return (train_indices, validation_indices, selected)


def evaluate_train_validation(args, model, loader, logger):
    model.eval()
    total_error = 0.0
    total_joints = 0
    with torch.no_grad():
        for batch_input, batch_gt in loader:
            batch_input = batch_input.cuda(non_blocking=True)
            batch_gt = batch_gt.cuda(non_blocking=True)
            if args.no_conf:
                batch_input = batch_input[:, :, :, :2]
            if args.flip:
                pred = (model(batch_input) + flip_data(model(flip_data(batch_input)))) / 2
            else:
                pred = model(batch_input)
            pred = pred - pred[:, :, 0:1, :]
            batch_gt = batch_gt - batch_gt[:, :, 0:1, :]
            per_joint_error = torch.norm(pred - batch_gt, dim=-1)
            total_error += per_joint_error.sum().item()
            total_joints += per_joint_error.numel()
    value = total_error / total_joints
    logger.info('Sequence-validation MPJPE (training coordinate scale): %.8f', value)
    return value
