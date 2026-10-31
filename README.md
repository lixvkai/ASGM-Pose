# ASGM-Pose

源码仓库：[lixvkai/ASGM-Pose](https://github.com/lixvkai/ASGM-Pose)。模型配置位于 `configs/h36m/`。

## 训练流程

使用 Python 3.10 及以上版本，并安装与本机 CUDA 环境兼容的 PyTorch 和 CUDA 工具包。在项目根目录执行：

```bash
pip install -r requirements.txt
pip install ./kernels/selective_scan --no-build-isolation
```

Human3.6M 原始数据可从[官方网站](http://vision.imar.ro/human3.6m/)注册获取；本流程使用的 Stacked Hourglass 预处理数据可从 [MotionBERT 数据说明页](https://github.com/Walter0807/MotionBERT/blob/main/docs/pose3d.md)中的下载链接获取。

将 `h36m_sh_conf_cam_source_final.pkl` 放入 `data/motion3d/`，然后生成 Human3.6M 数据片段：

```bash
python tools/prepare_h36m.py --output data/motion3d/MB3D_f243s81/H36M-SH
```

以 ASGM-Pose-B 为例，指定一个尚不存在的输出目录开始训练：

```bash
python train_h36m.py --config configs/h36m/ASGM_Pose_B.yaml --checkpoint checkpoints/h36m_B --seed 0 --split-seed 0 --heldout-sequences 2
```

1. **验证选轮数**：从训练受试者 S1、S5、S6、S7、S8 中，各留出两条完整序列作为验证集。同一序列的全部相机视角和 clips 保持在同一侧。使用验证集 MPJPE 选择训练轮数；YAML 中的 `epochs` 是该阶段的最大训练轮数。
2. **合并后重新训练**：将验证序列合并回训练集，重新初始化模型和优化器，按照选定轮数在完整训练集上训练。训练结束后，在 S9 和 S11 上进行最终测试。

默认使用 `--training-mode auto`，上述两个阶段自动衔接。可通过 `--epochs 120` 覆盖 YAML 中的最大验证训练轮数。

如果已经确定训练轮数，可选择单阶段模式，在完整训练集上直接训练指定轮数。例如训练 96 轮：

```bash
python train_h36m.py --config configs/h36m/ASGM_Pose_B.yaml --checkpoint checkpoints/h36m_B_96 --seed 0 --training-mode fixed --epochs 96
```

`fixed` 模式跳过验证选轮数，使用完整训练集训练，取最后一轮权重进行最终测试。`--epochs` 未传入时使用 YAML 的 `epochs`。该轮数应在最终测试前确定。

最终权重保存在 `checkpoints/h36m_B/best_epoch.bin`，自动两阶段模式的验证权重单独保存为 `best_validation_epoch.bin`。

训练 ASGM-Pose-S 或 ASGM-Pose-L 时，分别使用 `configs/h36m/ASGM_Pose_S.yaml` 或 `configs/h36m/ASGM_Pose_L.yaml`，并设置不同的输出目录。

验证阶段中断后，保持原有配置、随机种子、划分参数和输出目录不变，使用以下命令继续训练：

```bash
python train_h36m.py --config configs/h36m/ASGM_Pose_B.yaml --checkpoint checkpoints/h36m_B --seed 0 --split-seed 0 --heldout-sequences 2 --resume checkpoints/h36m_B/latest_validation_epoch.bin
```

如果在完整训练集重新训练阶段中断，将上述恢复文件改为 `latest_final_epoch.bin`。单阶段模式恢复时也使用 `latest_final_epoch.bin`，并保持原来的 `--training-mode fixed --epochs 96` 等参数。

## 推理流程

Human3.6M 预训练权重及对应配置：

| 模型 | 权重下载 | 配置文件 |
|---|---|---|
| ASGM-Pose-S | [Google Drive](https://drive.google.com/file/d/1P7mAYdCm1euVwCfIALQzL5Khtzanuaml/view?usp=drive_link) | `configs/h36m/ASGM_Pose_S.yaml` |
| ASGM-Pose-B | [Google Drive](https://drive.google.com/file/d/1hqlb0nukVdj8ZOd0gzMxuKrxU0UA2oR5/view?usp=drive_link) | `configs/h36m/ASGM_Pose_B.yaml` |
| ASGM-Pose-L | [Google Drive](https://drive.google.com/file/d/1pIsr9_0Bxjc_2HscaTEzdyNcYqdMSdhd/view?usp=drive_link) | `configs/h36m/ASGM_Pose_L.yaml` |

下载对应模型权重后，将下方命令的 `--checkpoint` 替换为实际文件路径。S/L 模型同时切换为表中对应的配置文件。

选择与已有权重匹配的模型配置进行评测：

```bash
python evaluate.py --config configs/h36m/ASGM_Pose_S.yaml --checkpoint checkpoints/h36m_S/best_epoch.bin
python evaluate.py --config configs/h36m/ASGM_Pose_B.yaml --checkpoint checkpoints/h36m_B/best_epoch.bin
python evaluate.py --config configs/h36m/ASGM_Pose_L.yaml --checkpoint checkpoints/h36m_L/best_epoch.bin
```
