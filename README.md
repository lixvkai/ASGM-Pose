# ASGM-Pose

This repository contains the Human3.6M training and evaluation code for ASGM-Pose. It intentionally includes only the data-preparation, training, and checkpoint-evaluation paths used by this release.

## Environment

Create a Python environment and install a PyTorch build compatible with your CUDA driver, then install the remaining dependencies:

```bash
conda create -n asgm-pose python=3.10
conda activate asgm-pose
# Install PyTorch and torchvision following https://pytorch.org/get-started/locally/
pip install -r requirements.txt
```

## Human3.6M data preparation

1. Download the MotionBERT-preprocessed Human3.6M file `h36m_sh_conf_cam_source_final.pkl` from the original PoseMamba data release:
   - [OneDrive](https://1drv.ms/u/s%21AvAdh0LSjEOlgU7BuUZcyafu8kzc?e=vobkjZ)
   - [Google Drive](https://drive.google.com/file/d/1WWoVAae7YKKKZpa1goO_7YcwVFNR528S/view?usp=sharing)

2. Place the downloaded file under `data/motion3d/`, generate clips, and copy the source file into the runtime data directory:

```bash
mkdir -p data/motion3d/MB3D_f243s81
python tools/convert_h36m.py
cp data/motion3d/h36m_sh_conf_cam_source_final.pkl \
   data/motion3d/MB3D_f243s81/
```

After preparation, `data/motion3d/MB3D_f243s81/` must contain `h36m_sh_conf_cam_source_final.pkl` and `H36M-SH/`. The repository ignores data files by default.

## Pretrained checkpoints

Download the Human3.6M checkpoint that matches the selected configuration:

| Model | Download |
| --- | --- |
| ASGM-Pose-S | [Google Drive](https://drive.google.com/file/d/1P7mAYdCm1euVwCfIALQzL5Khtzanuaml/view?usp=drive_link) |
| ASGM-Pose-B | [Google Drive](https://drive.google.com/file/d/1hqlb0nukVdj8ZOd0gzMxuKrxU0UA2oR5/view?usp=drive_link) |
| ASGM-Pose-L | [Google Drive](https://drive.google.com/file/d/1pIsr9_0Bxjc_2HscaTEzdyNcYqdMSdhd/view?usp=drive_link) |

Place the downloaded checkpoint at any local path; the evaluation command accepts the checkpoint path explicitly.

## Training

Train an S, B, or L configuration with a separate output directory:

```bash
python train.py --config configs/pose3d/ASGM_Pose_h36m_S.yaml --checkpoint checkpoint/ASGM_Pose_S
python train.py --config configs/pose3d/ASGM_Pose_h36m_B.yaml --checkpoint checkpoint/ASGM_Pose_B
python train.py --config configs/pose3d/ASGM_Pose_h36m_L.yaml --checkpoint checkpoint/ASGM_Pose_L
```

Training writes `best_epoch.bin` and `latest_epoch.bin` to the directory supplied by `--checkpoint`.

## Evaluation

Evaluate a trained or downloaded Human3.6M checkpoint as follows:

```bash
python train.py \
  --config configs/pose3d/ASGM_Pose_h36m_B.yaml \
  --evaluate /path/to/best_epoch.bin
```

Use the configuration that matches the checkpoint scale. This release provides benchmark evaluation from 2D keypoints to 3D poses; it does not include a separate in-the-wild video demo pipeline.

## License

This repository is derived from [PoseMamba](https://github.com/nankingjing/PoseMamba) and is distributed under the Apache License 2.0. See [LICENSE](LICENSE). Any redistribution must comply with that license and retain the required notices.
