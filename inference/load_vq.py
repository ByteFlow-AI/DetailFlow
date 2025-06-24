import torch
from pathlib import Path

from modelling.tokenizer import VQ_models
from tools import json_load, yaml_load
from modelling.detail_token import ResolutionDegradation


def load_vq_model(vq_model_path, ema=False, config_path=None, yaml_path=None, device=None, eval_mode=True, logger=None):
    
    checkpoint = torch.load(vq_model_path, map_location="cpu")
    if ema:
        model_state_dict = checkpoint["ema"]
        if logger is None:
            print(f"Load EMA vq model from {vq_model_path}")
        else:
            logger.info(f"Load EMA vq model from {vq_model_path}")
    else:
        model_state_dict = checkpoint["model"]

        if logger is None:
            print(f"Load vq model from {vq_model_path}")
        else:
            logger.info(f"Load vq model from {vq_model_path}")

    if config_path is None:
        config_path = Path(vq_model_path).parent.parent / "config.json"
    if yaml_path is None:
        yaml_path = Path(vq_model_path).parent.parent / "config.yaml"
    yaml_config = yaml_load(yaml_path)
    vq_type = yaml_config['vq_model']

    config = json_load(config_path)
    config.pop("encoder_ch_mult")
    config.pop("decoder_ch_mult")

    vq_model = VQ_models[vq_type](**config)
    if eval_mode:
        vq_model.eval()
        
    if device is not None:
        vq_model.to(device)
    vq_model.load_state_dict(model_state_dict)
    
    del checkpoint

    if yaml_config['content_degradation'] == 'resolution_power' and yaml_config['enc_type'] in ['siglip2']:
        max_image_size = yaml_config['max_image_size']
        res_deg = ResolutionDegradation(
            max_img_size=max_image_size, 
            max_token_num=yaml_config['num_latent_tokens'],
            group_size=yaml_config['group_size'],
            causal_num=config['causal_num'],
            p=yaml_config['degradation_power'], 
            enc_patch_size=config['enc_patch_size'], 
            dec_patch_size=config['dec_patch_size'],
            min_dec_res=config['dec_patch_size']
            )
    else:
        res_deg = None

    return vq_model, config, yaml_config, res_deg
