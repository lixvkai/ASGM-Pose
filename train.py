import os  # 用于文件和目录操作
import numpy as np  # 用于数组和数值运算
import argparse  # 用于解析命令行参数
import errno  # 用于处理文件系统错误
import math  # 数学运算
import pickle  # 用于序列化和反序列化数据
import datetime  # 处理日期和时间
import tensorboardX  # 用于记录训练日志并支持 TensorBoard 可视化
import torch.distributed  # PyTorch 分布式训练支持
from tqdm import tqdm  # 显示训练或测试进度条
import time  # 时间相关操作
import copy  # 深拷贝操作
import random  # 随机数生成
import prettytable  # 生成格式化表格输出
import yaml  # 读取和写入 YAML 配置文件
import torch  # PyTorch 深度学习框架
import torch.nn as nn  # 神经网络模块
import torch.nn.functional as F  # 功能函数
import torch.optim as optim  # 优化器
from torch.utils.data import DataLoader  # 数据加载器

import numpy
import torch.serialization

# 导入自定义模块
from lib.utils.tools import *  # 项目特定的工具函数
from lib.utils.learning import *  # 学习率调度等学习相关工具
from lib.utils.utils_data import flip_data  # 数据翻转函数
from lib.data.dataset_motion_2d import PoseTrackDataset2D, InstaVDataset2D  # 2D 姿态数据集
from lib.data.dataset_motion_3d import MotionDataset3D  # 3D 姿态数据集
from lib.data.augmentation import Augmenter2D  # 2D 数据增强
from lib.data.datareader_h36m import DataReaderH36M  # Human3.6M 数据读取器
from lib.model.loss import *  # 损失函数
import logger  # 日志记录模块
from logger import colorlogger  # 彩色日志记录器
from lib.model.loss import loss_spatial_rank, loss_temporal_rank

def parse_args():
    # 定义命令行参数解析函数
    parser = argparse.ArgumentParser()
    # 配置文件路径，默认为 configs/pretrain.yaml
    parser.add_argument("--config", type=str, default="configs/pretrain.yaml", help="Path to the config file.")
    # 检查点保存目录，默认为 checkpoint
    parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH', help='checkpoint directory')
    # 预训练模型目录，默认为 checkpoint
    parser.add_argument('-p', '--pretrained', default='checkpoint', type=str, metavar='PATH', help='pretrained checkpoint directory')
    # 继续训练的检查点文件名，默认为空
    parser.add_argument('-r', '--resume', default='', type=str, metavar='FILENAME', help='checkpoint to resume (file name)')
    # 评估的检查点文件名，默认为空
    parser.add_argument('-e', '--evaluate', default='', type=str, metavar='FILENAME', help='checkpoint to evaluate (file name)')
    # 用于微调的检查点文件名，默认为 latest_epoch.bin
    parser.add_argument('-ms', '--selection', default='latest_epoch.bin', type=str, metavar='FILENAME', help='checkpoint to finetune (file name)')
    # 随机种子，默认为 0
    parser.add_argument('-sd', '--seed', default=0, type=int, help='random seed')
    opts = parser.parse_args()  # 解析参数
    return opts

def set_random_seed(seed):
    # 设置随机种子以确保实验可重复性
    random.seed(seed)  # 设置 Python random 种子
    np.random.seed(seed)  # 设置 NumPy 随机种子
    torch.manual_seed(seed)  # 设置 PyTorch 随机种子

def save_checkpoint(chk_path, epoch, lr, optimizer, model_pos, min_loss):
    # 保存模型检查点
    log.info(f'Saving checkpoint to {chk_path}')  # 记录保存检查点的日志
    torch.save({
        'epoch': epoch + 1,  # 保存当前轮次（加 1 表示下一轮次）
        'lr': lr,  # 保存当前学习率
        'optimizer': optimizer.state_dict(),  # 保存优化器状态
        'model_pos': model_pos.state_dict(),  # 保存模型权重
        'min_loss': min_loss  # 保存当前最小验证损失
    }, chk_path)  # 使用 torch.save 保存检查点

def evaluate(args, model_pos, test_loader, datareader):
    # 在测试集上评估模型性能
    log.info('INFO: Testing')  # 记录测试开始日志
    results_all = []  # 存储所有预测结果
    model_pos.eval()  # 设置模型为评估模式
    with torch.no_grad():  # 禁用梯度计算以节省内存
        for batch_input, batch_gt in tqdm(test_loader):  # 遍历测试集
            N, T = batch_gt.shape[:2]  # 获取批次大小 N 和时间步 T
            if torch.cuda.is_available():  # 如果 GPU 可用
                batch_input = batch_input.cuda()  # 将输入移动到 GPU
            if args.no_conf:  # 如果忽略置信度
                batch_input = batch_input[:, :, :, :2]  # 仅保留 x, y 坐标
            if args.flip:  # 如果启用数据翻转增强
                batch_input_flip = flip_data(batch_input)  # 翻转输入数据
                predicted_3d_pos_1 = model_pos(batch_input)  # 原始数据预测
                predicted_3d_pos_flip = model_pos(batch_input_flip)  # 翻转数据预测
                predicted_3d_pos_2 = flip_data(predicted_3d_pos_flip)  # 翻转回原始方向
                predicted_3d_pos = (predicted_3d_pos_1 + predicted_3d_pos_2) / 2  # 取平均
            else:
                predicted_3d_pos = model_pos(batch_input)  # 直接预测
            if args.rootrel:  # 如果启用根关节相对归一化
                predicted_3d_pos[:, :, 0, :] = 0  # 将根关节坐标设为 0
            else:
                batch_gt[:, 0, 0, 2] = 0  # 将真实标签第一帧根关节深度设为 0
            if args.gt_2d:  # 如果使用真实 2D 坐标
                predicted_3d_pos[..., :2] = batch_input[..., :2]  # 替换预测的 x, y 坐标
            results_all.append(predicted_3d_pos.cpu().numpy())  # 保存预测结果到 CPU

    log.info(len(results_all))  # 记录预测结果数量（如 2228）
    results_all = np.concatenate(results_all)  # 拼接所有批次预测结果
    results_all = datareader.denormalize(results_all)  # 反归一化到原始坐标空间
    log.info(results_all.shape)  # 记录结果形状
    _, split_id_test = datareader.get_split_id()  # 获取测试集分割索引
    actions = np.array(datareader.dt_dataset['test']['action'])  # 测试集动作标签
    factors = np.array(datareader.dt_dataset['test']['2.5d_factor'])  # 2.5D 因子
    gts = np.array(datareader.dt_dataset['test']['joints_2.5d_image'])  # 真实 2.5D 关节坐标
    sources = np.array(datareader.dt_dataset['test']['source'])  # 数据来源

    num_test_frames = len(actions)  # 计算测试集总帧数
    log.info(f"num_test_frames:{num_test_frames}")  # 记录帧数（如 566920）
    frames = np.array(range(num_test_frames))  # 创建帧索引数组
    log.info(len(split_id_test))  # 记录分割片段数
    action_clips = actions[split_id_test]  # 提取动作片段
    factor_clips = factors[split_id_test]  # 提取因子片段
    source_clips = sources[split_id_test]  # 提取来源片段
    frame_clips = frames[split_id_test]  # 提取帧索引片段
    gt_clips = gts[split_id_test]  # 提取真实关节坐标片段
    assert len(results_all) == len(action_clips)  # 确保预测和片段数量一致

    e1_all = np.zeros(num_test_frames)  # 初始化 MPJPE 误差数组
    e2_all = np.zeros(num_test_frames)  # 初始化 P-MPJPE 误差数组
    oc = np.zeros(num_test_frames)  # 初始化计数数组
    results = {}  # 按动作存储 MPJPE 误差
    results_procrustes = {}  # 按动作存储 P-MPJPE 误差
    action_names = sorted(set(datareader.dt_dataset['test']['action']))  # 获取唯一动作名称
    for action in action_names:
        results[action] = []  # 初始化动作的 MPJPE 误差列表
        results_procrustes[action] = []  # 初始化动作的 P-MPJPE 误差列表
    block_list = ['s_09_act_05_subact_02', 's_09_act_10_subact_02', 's_09_act_13_subact_01']  # 无效数据片段列表

    for idx in range(len(action_clips)):  # 遍历每个测试片段
        source = source_clips[idx][0][:-6]  # 提取片段来源（去掉后 6 个字符）
        if source in block_list:  # 如果来源在无效列表中
            continue  # 跳过该片段
        frame_list = frame_clips[idx]  # 获取帧索引
        action = action_clips[idx][0]  # 获取动作名称
        factor = factor_clips[idx][:, None, None]  # 获取 2.5D 因子
        gt = gt_clips[idx]  # 获取真实关节坐标
        pred = results_all[idx]  # 获取预测结果
        pred *= factor  # 恢复预测结果到真实尺度

        pred = pred - pred[:, 0:1, :]  # 预测相对于根关节归一化
        gt = gt - gt[:, 0:1, :]  # 真实标签相对于根关节归一化
        err1 = mpjpe(pred, gt)  # 计算 MPJPE 误差
        err2 = p_mpjpe(pred, gt)  # 计算 P-MPJPE 误差
        e1_all[frame_list] += err1  # 累加 MPJPE 误差
        e2_all[frame_list] += err2  # 累加 P-MPJPE 误差
        oc[frame_list] += 1  # 增加帧计数

    for idx in range(num_test_frames):  # 遍历所有测试帧
        if e1_all[idx] > 0:  # 如果帧有有效数据
            err1 = e1_all[idx] / oc[idx]  # 计算平均 MPJPE 误差
            err2 = e2_all[idx] / oc[idx]  # 计算平均 P-MPJPE 误差
            action = actions[idx]  # 获取帧对应的动作
            results[action].append(err1)  # 添加到动作的 MPJPE 列表
            results_procrustes[action].append(err2)  # 添加到动作的 P-MPJPE 列表

    final_result = []  # 存储所有动作的平均 MPJPE
    final_result_procrustes = []  # 存储所有动作的平均 P-MPJPE
    summary_table = prettytable.PrettyTable()  # 创建表格
    summary_table.field_names = ['test_name'] + action_names  # 设置列名
    for action in action_names:  # 遍历动作
        final_result.append(np.mean(results[action]))  # 计算动作的平均 MPJPE
        final_result_procrustes.append(np.mean(results_procrustes[action]))  # 计算动作的平均 P-MPJPE
    summary_table.add_row(['P1'] + final_result)  # 添加 MPJPE 行
    summary_table.add_row(['P2'] + final_result_procrustes)  # 添加 P-MPJPE 行
    log.info(summary_table)  # 记录表格
    e1 = np.mean(np.array(final_result))  # 计算所有动作的平均 MPJPE
    e2 = np.mean(np.array(final_result_procrustes))  # 计算所有动作的平均 P-MPJPE
    log.info(f'Protocol #1 Error (MPJPE): {e1} mm')  # 记录 MPJPE
    log.info(f'Protocol #2 Error (P-MPJPE): {e2} mm')  # 记录 P-MPJPE
    log.info('----------')  # 记录分隔线
    return e1, e2, results_all  # 返回 MPJPE、P-MPJPE 和预测结果

def train_epoch(args, model_pos, train_loader, losses, optimizer, has_3d, has_gt):
    # 执行一个训练轮次的训练
    model_pos.train()  # 设置模型为训练模式
    for idx, (batch_input, batch_gt) in tqdm(enumerate(train_loader)):  # 遍历训练集
        batch_size = len(batch_input)  # 获取批次大小
        if torch.cuda.is_available():  # 如果 GPU 可用
            batch_input = batch_input.cuda()  # 将输入移动到 GPU
            batch_gt = batch_gt.cuda()  # 将真实标签移动到 GPU

        with torch.no_grad():  # 无梯度模式预处理数据
            if args.no_conf:  # 如果忽略置信度
                batch_input = batch_input[:, :, :, :2]  # 仅保留 x, y 坐标
            if not has_3d:  # 如果不是 3D 数据
                conf = copy.deepcopy(batch_input[:, :, :, 2:])  # 提取置信度
            if args.rootrel:  # 如果启用根关节相对归一化
                batch_gt = batch_gt - batch_gt[:, :, 0:1, :]  # 真实标签相对于根关节
            else:
                batch_gt[:, :, :, 2] = batch_gt[:, :, :, 2] - batch_gt[:, 0:1, 0:1, 2]  # 深度归一化
            if args.mask or args.noise:  # 如果启用数据增强
                batch_input = args.aug.augment2D(batch_input, noise=(args.noise and has_gt), mask=args.mask)  # 2D 数据增强

        predicted_3d_pos = model_pos(batch_input)  # 预测 3D 姿态
        optimizer.zero_grad()  # 清空优化器梯度

        if has_3d:  # 如果是 3D 数据
            loss_3d_pos = loss_mpjpe(predicted_3d_pos, batch_gt)  # MPJPE 损失
            loss_3d_scale = n_mpjpe(predicted_3d_pos, batch_gt)  # 归一化 MPJPE 损失
            loss_3d_velocity = loss_velocity(predicted_3d_pos, batch_gt)  # 速度一致性损失
            loss_lv = loss_limb_var(predicted_3d_pos)  # 肢体长度变化损失
            loss_lg = loss_limb_gt(predicted_3d_pos, batch_gt)  # 肢体长度匹配损失
            loss_a = loss_angle(predicted_3d_pos, batch_gt)  # 关节角度损失
            loss_av = loss_angle_velocity(predicted_3d_pos, batch_gt)  # 关节角度速度损失
            w_mpjpe = torch.tensor([1, 1, 2.5, 2.5, 1, 2.5, 2.5, 1, 1, 1, 1.5, 1.5, 4, 4, 1.5, 4, 4]).cuda()  # 关节权重
            loss_3d_w = weighted_mpjpe(predicted_3d_pos, batch_gt, w_mpjpe)  # 加权 MPJPE 损失

            # 新增rank损失
            loss_spatial = loss_spatial_rank(predicted_3d_pos, batch_gt)
            loss_temporal = loss_temporal_rank(predicted_3d_pos, batch_gt)

            dif_seq = predicted_3d_pos[:, 1:, :, :] - predicted_3d_pos[:, :-1, :, :]  # 计算相邻帧差异
            weights_joints = torch.ones_like(dif_seq).cuda()  # 初始化权重张量
            weights_mul = w_mpjpe  # 关节权重
            weights_joints = torch.mul(weights_joints.permute(0, 1, 3, 2), weights_mul).permute(0, 1, 3, 2)  # 加权
            loss_diff = torch.mean(torch.multiply(weights_joints, torch.square(dif_seq)))  # 时间序列差分损失

            loss_total = args.lambda_3d * loss_3d_pos + \
                         args.lambda_scale * loss_3d_scale + \
                         args.lambda_3d_velocity * loss_3d_velocity + \
                         args.lambda_lv * loss_lv + \
                         args.lambda_lg * loss_lg + \
                         args.lambda_a * loss_a + \
                         args.lambda_av * loss_av + \
                         args.lambda_3dw * loss_3d_w + \
                         args.lambda_diff * loss_diff + \
                         getattr(args, 'lambda_spatial_rank', 1.0) * loss_spatial + \
                         getattr(args, 'lambda_temporal_rank', 1.0) * loss_temporal  # 加权总损失

            losses['3d_pos'].update(loss_3d_pos.item(), batch_size)  # 更新 MPJPE 损失统计
            losses['3d_scale'].update(loss_3d_scale.item(), batch_size)  # 更新归一化 MPJPE 损失统计
            losses['3d_velocity'].update(loss_3d_velocity.item(), batch_size)  # 更新速度损失统计
            losses['lv'].update(loss_lv.item(), batch_size)  # 更新肢体长度变化损失统计
            losses['lg'].update(loss_lg.item(), batch_size)  # 更新肢体长度匹配损失统计
            losses['angle'].update(loss_a.item(), batch_size)  # 更新角度损失统计
            losses['angle_velocity'].update(loss_av.item(), batch_size)  # 更新角度速度损失统计
            losses['spatial_rank'] = losses.get('spatial_rank', AverageMeter())
            losses['temporal_rank'] = losses.get('temporal_rank', AverageMeter())
            losses['spatial_rank'].update(loss_spatial.item(), batch_size)
            losses['temporal_rank'].update(loss_temporal.item(), batch_size)
            losses['total'].update(loss_total.item(), batch_size)  # 更新总损失统计
        else:  # 如果是 2D 数据
            loss_2d_proj = loss_2d_weighted(predicted_3d_pos, batch_gt, conf)  # 2D 投影损失
            loss_total = loss_2d_proj  # 总损失为 2D 投影损失
            losses['2d_proj'].update(loss_2d_proj.item(), batch_size)  # 更新 2D 投影损失统计
            losses['total'].update(loss_total.item(), batch_size)  # 更新总损失统计

        loss_total.backward()  # 计算梯度
        # ==================== 新增修改 START ====================
        # 添加梯度裁剪 (Gradient Clipping)
        # 这是防止梯度爆炸和训练不稳定的关键步骤
        # max_norm=1.0 是一个常用的默认值，可以根据需要调整
        # torch.nn.utils.clip_grad_norm_(model_pos.parameters(), max_norm=1.0)# 只有多头版需要
        # ==================== 新增修改 END ======================
        optimizer.step()  # 更新模型参数

def get_beijing_timestamp():
    # 获取北京时间的时间戳
    local_offset = time.localtime().tm_gmtoff  # 获取本地 UTC 偏移量
    beijing_offset = int(8 * 60 * 60)  # 北京时间 UTC 偏移量（8小时）
    offset = local_offset - beijing_offset  # 计算偏移差
    timestamp = int(datetime.datetime.now().timestamp())  # 获取当前时间戳
    beijing_timestamp = timestamp - offset  # 调整为北京时间
    return beijing_timestamp

def train_with_config(args, opts):
    import torch
    # 根据配置和选项进行训练
    opts.checkpoint = opts.checkpoint + '_' + datetime.datetime.fromtimestamp(get_beijing_timestamp()).strftime('%Y_%m_%d_T_%H_%M_%S')  # 检查点目录附加北京时间戳
    global log
    log = colorlogger(opts.checkpoint, log_name='log.txt')  # 初始化彩色日志记录器
    log.info(args)  # 记录配置参数
    with open(os.path.join(opts.checkpoint, 'config.yaml'), 'w') as f:  # 保存配置到 YAML 文件
        yaml.dump(args, f, sort_keys=False)
    log.info(f"Number of GPUs found: {torch.cuda.device_count()}")  # 记录可用 GPU 数量

    try:
        os.makedirs(opts.checkpoint)  # 创建检查点目录
    except OSError as e:
        if e.errno != errno.EEXIST:  # 如果目录已存在则忽略
            raise RuntimeError('Unable to create checkpoint directory:', opts.checkpoint)

    train_writer = tensorboardX.SummaryWriter(os.path.join(opts.checkpoint, "logs"))  # 初始化 TensorBoard 日志记录器
    log.info('Loading dataset...')  # 记录加载数据集日志

    trainloader_params = {  # 训练 DataLoader 参数
        'batch_size': args.batch_size,  # 批次大小
        'shuffle': True,  # 打乱数据
        'num_workers': 12,  # 数据加载子进程数
        'pin_memory': True,  # 内存锁定加速 GPU 传输
        'prefetch_factor': 4,  # 预取因子
        'persistent_workers': True  # 保持工作进程存活
    }
    testloader_params = {  # 测试 DataLoader 参数
        'batch_size': args.batch_size,
        'shuffle': False,  # 不打乱数据
        'num_workers': 12,
        'pin_memory': True,
        'prefetch_factor': 4,
        'persistent_workers': True
    }

    train_dataset = MotionDataset3D(args, args.subset_list, 'train')  # 创建 3D 训练数据集
    test_dataset = MotionDataset3D(args, args.subset_list, 'test')  # 创建 3D 测试数据集
    train_loader_3d = DataLoader(train_dataset, **trainloader_params)  # 创建训练 DataLoader
    test_loader = DataLoader(test_dataset, **testloader_params)  # 创建测试 DataLoader

    if args.train_2d:  # 如果启用 2D 训练
        posetrack = PoseTrackDataset2D()  # 创建 PoseTrack 2D 数据集
        posetrack_loader_2d = DataLoader(posetrack, **trainloader_params)  # 创建 PoseTrack DataLoader
        instav = InstaVDataset2D()  # 创建 InstaV 2D 数据集
        instav_loader_2d = DataLoader(instav, **trainloader_params)  # 创建 InstaV DataLoader

    datareader = DataReaderH36M(n_frames=args.clip_len, sample_stride=args.sample_stride, data_stride_train=args.data_stride, data_stride_test=args.clip_len, dt_root=args.data_root, dt_file=args.dt_file)  # 创建 Human3.6M 数据读取器
    min_loss = 100000  # 初始化最小验证损失

    model_backbone = load_backbone(args)  # 加载模型骨干网络
    model_params = 0  # 初始化参数计数
    for parameter in model_backbone.parameters():  # 遍历模型参数
        model_params += parameter.numel()  # 累加参数数量
    log.info(f'INFO: Trainable parameter count: {model_params}')  # 记录可训练参数数量

    if torch.cuda.is_available():  # 如果 GPU 可用
        model_backbone = nn.DataParallel(model_backbone)  # 使用 DataParallel 进行多 GPU 并行
        model_backbone = model_backbone.cuda()  # 将模型移动到 GPU

    if args.finetune:  # 如果启用微调
        if opts.resume or opts.evaluate:  # 如果继续训练或评估
            chk_filename = opts.evaluate if opts.evaluate else opts.resume  # 选择检查点文件
            log.info(f'Loading checkpoint {chk_filename}')  # 记录加载检查点日志
            checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)  # 加载检查点
            model_backbone.load_state_dict(checkpoint['model_pos'], strict=True)  # 加载模型权重
            model_pos = model_backbone
        else:
            chk_filename = os.path.join(opts.pretrained, opts.selection)  # 使用预训练检查点
            log.info(f'Loading checkpoint {chk_filename}')
            checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
            model_backbone.load_state_dict(checkpoint['model_pos'], strict=True)
            model_pos = model_backbone
    else:
        chk_filename = os.path.join(opts.checkpoint, "latest_epoch.bin")  # 默认最新检查点
        if os.path.exists(chk_filename):
            opts.resume = chk_filename
        if opts.resume or opts.evaluate:
            chk_filename = opts.evaluate if opts.evaluate else opts.resume
            log.info(f'Loading checkpoint {chk_filename}')
            import torch.serialization
            torch.serialization.add_safe_globals([numpy._core.multiarray.scalar])
            checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage, weights_only=False)
            if args.backbone == 'MotionAGFormer':
                model_backbone.load_state_dict(checkpoint['model'], strict=True)
            else:
                model_backbone.load_state_dict(checkpoint['model_pos'], strict=True)
        model_pos = model_backbone

    if args.partial_train:  # 如果启用部分训练
        model_pos = partial_train_layers(model_pos, args.partial_train)  # 设置部分层可训练

    if not opts.evaluate:  # 如果不是仅评估
        lr = args.learning_rate  # 初始化学习率
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model_pos.parameters()), lr=lr, weight_decay=args.weight_decay)  # 创建 AdamW 优化器
        lr_decay = args.lr_decay  # 学习率衰减因子
        st = 0  # 初始化起始轮次
        if args.train_2d:  # 如果启用 2D 训练
            log.info(f'INFO: Training on {len(train_loader_3d)}(3D)+{len(instav_loader_2d) + len(posetrack_loader_2d)}(2D) batches')  # 记录训练批次信息
        else:
            log.info(f'INFO: Training on {len(train_loader_3d)}(3D) batches')
        if opts.resume:  # 如果继续训练
            st = checkpoint['epoch']  # 从检查点恢复轮次
            if 'optimizer' in checkpoint and checkpoint['optimizer'] is not None:
                optimizer.load_state_dict(checkpoint['optimizer'])  # 恢复优化器状态
            else:
                log.info('WARNING: this checkpoint does not contain an optimizer state. The optimizer will be reinitialized.')
            lr = checkpoint['lr']  # 恢复学习率
            if 'min_loss' in checkpoint and checkpoint['min_loss'] is not None:
                min_loss = checkpoint['min_loss']  # 恢复最小损失

        args.mask = (args.mask_ratio > 0 and args.mask_T_ratio > 0)  # 判断是否启用掩码增强
        if args.mask or args.noise:  # 如果启用掩码或噪声增强
            args.aug = Augmenter2D(args)  # 创建 2D 数据增强器

        for epoch in range(st, args.epochs):  # 遍历训练轮次
            log.info(f'Training epoch {epoch}.')  # 记录当前轮次
            start_time = time.time()  # 记录开始时间
            losses = {  # 初始化损失统计字典
                '3d_pos': AverageMeter(),
                '3d_scale': AverageMeter(),
                '2d_proj': AverageMeter(),
                'lg': AverageMeter(),
                'lv': AverageMeter(),
                'total': AverageMeter(),
                '3d_velocity': AverageMeter(),
                'angle': AverageMeter(),
                'angle_velocity': AverageMeter()
            }
            if args.train_2d and (epoch >= args.pretrain_3d_curriculum):  # 如果启用 2D 训练且达到课程学习阶段
                train_epoch(args, model_pos, posetrack_loader_2d, losses, optimizer, has_3d=False, has_gt=True)  # 训练 PoseTrack 2D 数据
                train_epoch(args, model_pos, instav_loader_2d, losses, optimizer, has_3d=False, has_gt=False)  # 训练 InstaV 2D 数据
            train_epoch(args, model_pos, train_loader_3d, losses, optimizer, has_3d=True, has_gt=True)  # 训练 3D 数据
            elapsed = (time.time() - start_time) / 60  # 计算轮次耗时（分钟）

            if args.no_eval:  # 如果不进行评估
                log.info('[%d] time %.2f lr %f 3d_train %f' % (
                    epoch + 1, elapsed, lr, losses['3d_pos'].avg))  # 记录训练信息
            else:
                e1, e2, results_all = evaluate(args, model_pos, test_loader, datareader)  # 评估模型
                log.info('[%d] time %.2f lr %f 3d_train %f e1 %f e2 %f' % (
                    epoch + 1, elapsed, lr, losses['3d_pos'].avg, e1, e2))  # 记录训练和评估信息
                log.info(f'Remaining training time: {datetime.timedelta(seconds=(time.time() - start_time) * (args.epochs - epoch))}')  # 记录剩余训练时间
                train_writer.add_scalar('Error P1', e1, epoch + 1)  # 记录 MPJPE 到 TensorBoard
                train_writer.add_scalar('Error P2', e2, epoch + 1)  # 记录 P-MPJPE 到 TensorBoard
                train_writer.add_scalar('loss_3d_pos', losses['3d_pos'].avg, epoch + 1)  # 记录损失到 TensorBoard
                train_writer.add_scalar('loss_2d_proj', losses['2d_proj'].avg, epoch + 1)
                train_writer.add_scalar('loss_3d_scale', losses['3d_scale'].avg, epoch + 1)
                train_writer.add_scalar('loss_3d_velocity', losses['3d_velocity'].avg, epoch + 1)
                train_writer.add_scalar('loss_lv', losses['lv'].avg, epoch + 1)
                train_writer.add_scalar('loss_lg', losses['lg'].avg, epoch + 1)
                train_writer.add_scalar('loss_a', losses['angle'].avg, epoch + 1)
                train_writer.add_scalar('loss_av', losses['angle_velocity'].avg, epoch + 1)
                train_writer.add_scalar('loss_total', losses['total'].avg, epoch + 1)

            lr *= lr_decay  # 指数衰减学习率
            for param_group in optimizer.param_groups:
                param_group['lr'] *= lr_decay  # 更新优化器学习率

            chk_path = os.path.join(opts.checkpoint, f'epoch_{epoch}.bin')  # 轮次检查点路径
            chk_path_latest = os.path.join(opts.checkpoint, 'latest_epoch.bin')  # 最新检查点路径
            chk_path_best = os.path.join(opts.checkpoint, 'best_epoch.bin')  # 最佳检查点路径
            save_checkpoint(chk_path_latest, epoch, lr, optimizer, model_pos, min_loss)  # 保存最新检查点
            if (epoch + 1) % args.checkpoint_frequency == 0:  # 每隔指定轮次保存检查点
                save_checkpoint(chk_path, epoch, lr, optimizer, model_pos, min_loss)
            if e1 < min_loss:  # 如果当前 MPJPE 小于最小损失
                min_loss = e1  # 更新最小损失
                save_checkpoint(chk_path_best, epoch, lr, optimizer, model_pos, min_loss)  # 保存最佳检查点

    if opts.evaluate:  # 如果仅评估
        e1, e2, results_all = evaluate(args, model_pos, test_loader, datareader)  # 评估模型

if __name__ == "__main__":
    opts = parse_args()  # 解析命令行参数
    set_random_seed(opts.seed)  # 设置随机种子
    args = get_config(opts.config)  # 获取配置文件参数
    train_with_config(args, opts)  # 执行训练