from dataclasses import dataclass, field
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from modelling.modules.siglip2_vit import CustomSiglip2VisionModel, SiglipEncoder, SiglipDecoder


def build_mlp(hidden_size, projector_dim, z_dim):
    return nn.Sequential(
                nn.Linear(hidden_size, projector_dim),
                nn.SiLU(),
                nn.Linear(projector_dim, projector_dim),
                nn.SiLU(),
                nn.Linear(projector_dim, z_dim),
            )


def build_encoder(config, num_latent_tokens, group_size):
    if config.enc_type == 'siglip2':
        if config.encoder_model == "siglip2_base":
            encoder_cfg = {'layer_num': 12, 'embed_dim':768, 'num_heads': 12}
            if config.enc_pretrained:
                encoder_cfg = {'layer_num': 12}
        elif config.encoder_model == "siglip2_large":
            encoder_cfg = {'layer_num': 16}
        elif config.encoder_model == "siglip2":
            encoder_cfg = {'layer_num': None}
        else:
            raise NotImplementedError()
        encoder = SiglipEncoder(
            pretrained=config.enc_pretrained,
            num_latent_tokens=num_latent_tokens,
            patch_size=config.enc_patch_size,
            causal_mode=config.causal_encoder,
            group_size=group_size,
            causal_num=config.causal_num,
            **encoder_cfg
        )
    else:
        raise NotImplementedError()
    
    return encoder


def build_decoder(config, num_latent_tokens, group_size, repa_layer_indices):
    if config.dec_type == 'siglip2':
        if config.decoder_model == "siglip2_base":
            decoder_cfg = {'layer_num': 12, 'embed_dim':768, 'num_heads': 12}
            if config.dec_pretrained:
                decoder_cfg = {'layer_num': 12}
        elif config.decoder_model == "siglip2_large":
            decoder_cfg = {'layer_num': 16}
        elif config.decoder_model == "siglip2":
            decoder_cfg = {'layer_num': None}
        decoder = SiglipDecoder(
            pretrained=config.dec_pretrained,
            num_latent_tokens=num_latent_tokens,
            patch_size=config.dec_patch_size,
            causal_mode=config.causal_decoder,
            max_image_size=config.max_image_size,
            group_size=group_size,
            repa_layer=repa_layer_indices,
            causal_num=config.causal_num,
            **decoder_cfg,
            )
    else:
        raise NotImplementedError()
    
    return decoder


def build_repa_model(config, embed_dim):
    if config.repa_model == 'siglip2':
        repa_model = CustomSiglip2VisionModel(pretrained=True, layer_num=None, patch_size=config.repa_patch_size)
        if config.repa_patch_size != config.dec_patch_size or config.repa_patch_size != repa_model.raw_patch_size:
            raise NotImplementedError()
        repa_preprocessor = nn.Identity()
    else:
        raise NotImplementedError()
    
    repa_z_dim = repa_model.embed_dim
    projection = build_mlp(embed_dim, config.repa_proj_dim, repa_z_dim)

    for param in repa_model.parameters():
        param.requires_grad = False
    repa_model.eval()

    return repa_preprocessor, repa_model, projection