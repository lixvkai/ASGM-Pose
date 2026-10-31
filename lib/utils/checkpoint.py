import torch

def read_checkpoint(path):
    return torch.load(path, map_location='cpu', weights_only=False)

def load_model_weights(model, checkpoint):
    state = checkpoint.get('model_pos', checkpoint.get('model', checkpoint.get('state_dict', checkpoint)))
    state = {k.removeprefix('module.'): v for k, v in state.items()}
    target = model.module if hasattr(model, 'module') else model
    return target.load_state_dict(state, strict=True)
