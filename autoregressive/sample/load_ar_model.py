import torch
from pathlib import Path

from autoregressive.models.gpt import GPT_models
from tools import json_load, yaml_load


def load_ar_model(gpt_model_path, ema=False, config_path=None, yaml_path=None, device=None, eval_mode=True, dtype=None, from_fsdp=False):
    
    checkpoint = torch.load(gpt_model_path, map_location="cpu")
    if ema:
        model_state_dict = checkpoint["ema"]
    elif from_fsdp:
        model_state_dict = checkpoint
    elif "model" in checkpoint:  # ddp
        model_state_dict = checkpoint["model"]
    elif "module" in checkpoint: # deepspeed
        model_state_dict = checkpoint["module"]
    elif "state_dict" in checkpoint:
        model_state_dict = checkpoint["state_dict"]
    else:
        print("Model loading failure")
        exit()

    if config_path is None:
        config_path = Path(gpt_model_path).parent.parent / "config.json"
    if yaml_path is None:
        yaml_path = Path(gpt_model_path).parent.parent / "config.yaml"
    yaml_config = yaml_load(yaml_path)
    config = json_load(config_path)
    config.pop('n_layer')
    config.pop('n_head')
    config.pop('dim')

    ar_model = GPT_models[yaml_config['gpt_model']](**config)
    if eval_mode:
        ar_model.eval()
        
    if device is not None:
        ar_model.to(device)
    ar_model.load_state_dict(model_state_dict)
    if dtype is not None:
        ar_model.to(dtype=dtype)
    
    del checkpoint
    return ar_model, config, yaml_config