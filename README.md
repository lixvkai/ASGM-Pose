# ASGM-Pose

This repository releases the core network implementation of **ASGM-Pose** for monocular 3D human pose estimation. It contains the heterogeneous spatial graph module, temporal state-space blocks, token selection/restoration pathway, and their custom scan operators.

The release intentionally excludes dataset preparation, training, evaluation, configuration files, checkpoints, and demo code.

## Dependencies

Use Python 3.10 or later. Install a PyTorch build compatible with the local CUDA environment following the [official PyTorch instructions](https://pytorch.org/get-started/locally/), then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

## Model interface

From the repository root, instantiate the model as follows:

```python
import torch

from lib.model.ASGM_Pose import ASGM_Pose

model = ASGM_Pose(
    num_frame=243,
    num_joints=17,
    in_chans=2,
    embed_dim_ratio=128,
    depth=10,
    mlp_ratio=2.0,
    token_num=81,
    layer_index=5,
).cuda().eval()

x = torch.randn(1, 243, 17, 2, device="cuda")
with torch.no_grad():
    y = model(x)
```

The input tensor has shape `(B, T, J, C)`, where `B`, `T`, `J`, and `C` denote batch size, temporal length, number of joints, and input channels, respectively. The default input is 2D joint coordinates (`C=2`), and the model outputs 3D joint coordinates with shape `(B, T, J, 3)`.

## Core files

- `lib/model/ASGM_Pose.py`: ASGM-Pose architecture and adaptive graph convolution.
- `lib/model/mambablocks.py`: temporal state-space blocks.
- `lib/model/csms6s.py` and `lib/model/csm_triton.py`: selective-scan operators.
- `lib/model/drop.py`: stochastic-depth layer.

## License

This repository is derived from [PoseMamba](https://github.com/nankingjing/PoseMamba) and is distributed under the Apache License 2.0. See [LICENSE](LICENSE).
