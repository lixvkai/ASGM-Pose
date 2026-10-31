import numpy as np
from tqdm import tqdm
import copy
import prettytable
import torch


from lib.utils.learning import AverageMeter
from lib.utils.utils_data import flip_data
from lib.model.loss import (
    loss_2d_weighted,
    loss_angle,
    loss_angle_velocity,
    loss_limb_gt,
    loss_limb_var,
    loss_mpjpe,
    loss_spatial_rank,
    loss_temporal_rank,
    loss_velocity,
    mpjpe,
    n_mpjpe,
    p_mpjpe,
    weighted_mpjpe,
)


def evaluate(args, model_pos, test_loader, datareader):

    log.info('INFO: Testing')
    results_all = []
    model_pos.eval()
    with torch.no_grad():
        for batch_input, batch_gt in tqdm(test_loader):
            N, T = batch_gt.shape[:2]
            if torch.cuda.is_available():
                batch_input = batch_input.cuda()
            if args.no_conf:
                batch_input = batch_input[:, :, :, :2]
            if args.flip:
                batch_input_flip = flip_data(batch_input)
                predicted_3d_pos_1 = model_pos(batch_input)
                predicted_3d_pos_flip = model_pos(batch_input_flip)
                predicted_3d_pos_2 = flip_data(predicted_3d_pos_flip)
                predicted_3d_pos = (predicted_3d_pos_1 + predicted_3d_pos_2) / 2
            else:
                predicted_3d_pos = model_pos(batch_input)
            if args.rootrel:
                predicted_3d_pos[:, :, 0, :] = 0
            else:
                batch_gt[:, 0, 0, 2] = 0
            if args.gt_2d:
                predicted_3d_pos[..., :2] = batch_input[..., :2]
            results_all.append(predicted_3d_pos.cpu().numpy())

    log.info(len(results_all))
    results_all = np.concatenate(results_all)
    results_all = datareader.denormalize(results_all)
    log.info(results_all.shape)
    _, split_id_test = datareader.get_split_id()
    actions = np.array(datareader.dt_dataset['test']['action'])
    factors = np.array(datareader.dt_dataset['test']['2.5d_factor'])
    gts = np.array(datareader.dt_dataset['test']['joints_2.5d_image'])
    sources = np.array(datareader.dt_dataset['test']['source'])

    num_test_frames = len(actions)
    log.info(f"num_test_frames:{num_test_frames}")
    frames = np.array(range(num_test_frames))
    log.info(len(split_id_test))
    action_clips = actions[split_id_test]
    factor_clips = factors[split_id_test]
    source_clips = sources[split_id_test]
    frame_clips = frames[split_id_test]
    gt_clips = gts[split_id_test]
    assert len(results_all) == len(action_clips)

    e1_all = np.zeros(num_test_frames)
    e2_all = np.zeros(num_test_frames)
    oc = np.zeros(num_test_frames)
    results = {}
    results_procrustes = {}
    action_names = sorted(set(datareader.dt_dataset['test']['action']))
    for action in action_names:
        results[action] = []
        results_procrustes[action] = []

    block_list = ['s_09_act_05_subact_02', 's_09_act_10_subact_02', 's_09_act_13_subact_01']

    for idx in range(len(action_clips)):
        source = source_clips[idx][0][:-6]
        if source in block_list:
            continue
        frame_list = frame_clips[idx]
        action = action_clips[idx][0]
        factor = factor_clips[idx][:, None, None]
        gt = gt_clips[idx]
        pred = results_all[idx]
        pred *= factor

        pred = pred - pred[:, 0:1, :]
        gt = gt - gt[:, 0:1, :]
        err1 = mpjpe(pred, gt)
        err2 = p_mpjpe(pred, gt)
        e1_all[frame_list] += err1
        e2_all[frame_list] += err2
        oc[frame_list] += 1

    for idx in range(num_test_frames):
        if e1_all[idx] > 0:
            err1 = e1_all[idx] / oc[idx]
            err2 = e2_all[idx] / oc[idx]
            action = actions[idx]
            results[action].append(err1)
            results_procrustes[action].append(err2)

    final_result = []
    final_result_procrustes = []
    summary_table = prettytable.PrettyTable()
    summary_table.field_names = ['test_name'] + action_names
    for action in action_names:
        final_result.append(np.mean(results[action]))
        final_result_procrustes.append(np.mean(results_procrustes[action]))
    summary_table.add_row(['P1'] + final_result)
    summary_table.add_row(['P2'] + final_result_procrustes)
    log.info(summary_table)
    e1 = np.mean(np.array(final_result))
    e2 = np.mean(np.array(final_result_procrustes))
    log.info(f'Protocol #1 Error (MPJPE): {e1} mm')
    log.info(f'Protocol #2 Error (P-MPJPE): {e2} mm')
    log.info('----------')
    return e1, e2, results_all

def train_epoch(args, model_pos, train_loader, losses, optimizer, has_3d, has_gt):

    model_pos.train()
    for idx, (batch_input, batch_gt) in tqdm(enumerate(train_loader)):
        batch_size = len(batch_input)
        if torch.cuda.is_available():
            batch_input = batch_input.cuda()
            batch_gt = batch_gt.cuda()

        with torch.no_grad():
            if args.no_conf:
                batch_input = batch_input[:, :, :, :2]
            if not has_3d:
                conf = copy.deepcopy(batch_input[:, :, :, 2:])
            if args.rootrel:
                batch_gt = batch_gt - batch_gt[:, :, 0:1, :]
            else:
                batch_gt[:, :, :, 2] = batch_gt[:, :, :, 2] - batch_gt[:, 0:1, 0:1, 2]
            if args.mask or args.noise:
                batch_input = args.aug.augment2D(batch_input, noise=(args.noise and has_gt), mask=args.mask)

        predicted_3d_pos = model_pos(batch_input)
        optimizer.zero_grad()

        if has_3d:
            loss_3d_pos = loss_mpjpe(predicted_3d_pos, batch_gt)
            loss_3d_scale = n_mpjpe(predicted_3d_pos, batch_gt)
            loss_3d_velocity = loss_velocity(predicted_3d_pos, batch_gt)
            loss_lv = loss_limb_var(predicted_3d_pos)
            loss_lg = loss_limb_gt(predicted_3d_pos, batch_gt)
            loss_a = loss_angle(predicted_3d_pos, batch_gt)
            loss_av = loss_angle_velocity(predicted_3d_pos, batch_gt)
            w_mpjpe = torch.tensor([1, 1, 2.5, 2.5, 1, 2.5, 2.5, 1, 1, 1, 1.5, 1.5, 4, 4, 1.5, 4, 4]).cuda()
            loss_3d_w = weighted_mpjpe(predicted_3d_pos, batch_gt, w_mpjpe)


            loss_spatial = loss_spatial_rank(predicted_3d_pos, batch_gt)
            loss_temporal = loss_temporal_rank(predicted_3d_pos, batch_gt)

            dif_seq = predicted_3d_pos[:, 1:, :, :] - predicted_3d_pos[:, :-1, :, :]
            weights_joints = torch.ones_like(dif_seq).cuda()
            weights_mul = w_mpjpe
            weights_joints = torch.mul(weights_joints.permute(0, 1, 3, 2), weights_mul).permute(0, 1, 3, 2)
            loss_diff = torch.mean(torch.multiply(weights_joints, torch.square(dif_seq)))

            loss_total = args.lambda_3d * loss_3d_pos +\
                         args.lambda_scale * loss_3d_scale +\
                         args.lambda_3d_velocity * loss_3d_velocity +\
                         args.lambda_lv * loss_lv +\
                         args.lambda_lg * loss_lg +\
                         args.lambda_a * loss_a +\
                         args.lambda_av * loss_av +\
                         args.lambda_3dw * loss_3d_w +\
                         args.lambda_diff * loss_diff +\
                         getattr(args, 'lambda_spatial_rank', 1.0) * loss_spatial +\
                         getattr(args, 'lambda_temporal_rank', 1.0) * loss_temporal

            losses['3d_pos'].update(loss_3d_pos.item(), batch_size)
            losses['3d_scale'].update(loss_3d_scale.item(), batch_size)
            losses['3d_velocity'].update(loss_3d_velocity.item(), batch_size)
            losses['lv'].update(loss_lv.item(), batch_size)
            losses['lg'].update(loss_lg.item(), batch_size)
            losses['angle'].update(loss_a.item(), batch_size)
            losses['angle_velocity'].update(loss_av.item(), batch_size)
            losses['spatial_rank'] = losses.get('spatial_rank', AverageMeter())
            losses['temporal_rank'] = losses.get('temporal_rank', AverageMeter())
            losses['spatial_rank'].update(loss_spatial.item(), batch_size)
            losses['temporal_rank'].update(loss_temporal.item(), batch_size)
            losses['total'].update(loss_total.item(), batch_size)
        else:
            loss_2d_proj = loss_2d_weighted(predicted_3d_pos, batch_gt, conf)
            loss_total = loss_2d_proj
            losses['2d_proj'].update(loss_2d_proj.item(), batch_size)
            losses['total'].update(loss_total.item(), batch_size)

        loss_total.backward()


        optimizer.step()


if __name__ == "__main__":
    from lib.training.h36m import main
    main(train_epoch, evaluate)
