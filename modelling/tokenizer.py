# Modified from:
#   taming-transformers: https://github.com/CompVis/taming-transformers
#   maskgit: https://github.com/google-research/maskgit
#   SoftVQ-VAE: https://github.com/Hhhhhhao/continuous_tokenizer/blob/main/modelling/tokenizer.py
from dataclasses import dataclass, field
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin
from torch.utils.checkpoint import checkpoint
from tools import json_dump
from einops import rearrange

from modelling.quantizers.vq import VectorQuantizer
from modelling.quantizers.kl import DiagonalGaussianDistribution
from modelling.quantizers.softvq import SoftVectorQuantizer
from modelling.detail_token import average_cosine_similarity
from modelling.model_builder import build_encoder, build_decoder, build_repa_model, build_mlp


def mean_flat(x):
    """
    Take the mean over all non-batch dimensions.
    """
    return torch.mean(x, dim=list(range(1, len(x.size()))))


@dataclass
class ModelArgs:
    image_size: int = 256
    max_image_size: int = 256
    codebook_size: int = 16384
    codebook_embed_dim: int = 8
    codebook_l2_norm: bool = True
    codebook_show_usage: bool = True
    commit_loss_beta: float = 0.25
    entropy_loss_ratio: float = 0.0
    vq_loss_ratio: float = 1.0 # for soft vq
    kl_loss_weight: float = 0.000001
    tau: float = 0.1
    num_codebooks: int = 1
    
    encoder_ch_mult: List[int] = field(default_factory=lambda: [1, 1, 2, 2, 4])
    decoder_ch_mult: List[int] = field(default_factory=lambda: [1, 1, 2, 2, 4])
    z_channels: int = 256
    dropout_p: float = 0.0

    enc_type: str = 'cnn'
    dec_type: str = 'cnn'
    encoder_model: str = 'llamagen_encoder'
    decoder_model: str = 'llamagen_decoder'
    num_latent_tokens: int = 256
    
    # for pre-trained models
    enc_tuning_method: str = 'full'
    dec_tuning_method: str = 'full'
    enc_pretrained: bool = True
    dec_pretrained: bool = False 
    
    # for vit 
    enc_patch_size: int = 16
    dec_patch_size: int = 16
    enc_drop_path_rate: float = 0.0
    dec_drop_path_rate: float = 0.0
    
    # repa for vit
    repa: bool = False
    repa_patch_size: int = 16
    repa_model: str = 'vit_base_patch16_224'
    repa_proj_dim: int = 2048
    repa_layer_indices: int = 1
    repa_loss_weight: float = 0.1
    repa_align: str = 'global'
    
    vq_mean: float = 0.0
    vq_std: float = 1.0

    causal_encoder: bool = False 
    causal_decoder: bool = True 
    gradient_checkpointing_encoder: bool = False 
    gradient_checkpointing_decoder: bool = False
    group_size: int = 1
    causal_num: int = None
    global_token_loss_weight: float = 0.0
    correction_training: bool = True 
    

class VQModel(nn.Module, PyTorchModelHubMixin):
    def __init__(self, config: ModelArgs, 
                tags=["arxiv:2505.21473", "image-generation", "128 tokens", "DetailFlow"], 
                repo_url="https://github.com/ByteFlow-AI/DetailFlow", 
                license="apache-2.0"):
        super().__init__()
        self.config = config
        self.vq_mean = config.vq_mean
        self.vq_std = config.vq_std
        self.num_latent_tokens = config.num_latent_tokens
        self.codebook_embed_dim = config.codebook_embed_dim
        self.group_size = config.group_size
        self.correction_training = config.correction_training
        self.causal_num = self.group_size if config.causal_num is None else config.causal_num
        
        self.repa = config.repa
        self.repa_loss_weight = config.repa_loss_weight
        self.repa_align = config.repa_align
        self.global_token_learning = config.global_token_loss_weight != 0
        self.global_token_loss_weight = config.global_token_loss_weight
        
        self.gradient_checkpointing_encoder = config.gradient_checkpointing_encoder
        self.gradient_checkpointing_decoder = config.gradient_checkpointing_decoder
        
        self.encoder = build_encoder(config=config, num_latent_tokens=self.num_latent_tokens, group_size=self.group_size)

        repa_layer_indices = config.repa_layer_indices if self.repa else None
        self.decoder = build_decoder(config=config, num_latent_tokens=self.num_latent_tokens, group_size=self.group_size, repa_layer_indices=repa_layer_indices)
        
        self.quant_conv = nn.Linear(self.encoder.embed_dim, config.codebook_embed_dim)
        self.post_quant_conv = nn.Linear(config.codebook_embed_dim, self.decoder.embed_dim)
        if self.correction_training:
            self.refine_conv = nn.Linear(config.codebook_embed_dim, self.encoder.embed_dim)
        
        if self.repa or self.global_token_learning:
            self.repa_preprocessor, self.repa_model, self.projection = build_repa_model(config=config, embed_dim=self.decoder.embed_dim)
            if not self.repa:
                self.projection = nn.Identity()
            if self.global_token_learning:
                self.global_token_proj = build_mlp(config.codebook_embed_dim, 1024, self.repa_model.embed_dim)
        
        self.quantize = VectorQuantizer(config.codebook_size, config.codebook_embed_dim, 
                                        config.commit_loss_beta, config.entropy_loss_ratio,
                                        config.codebook_l2_norm, config.codebook_show_usage, config.vq_loss_ratio)
    
    def encode_image(self, x, cur_max_token_num=None, noise_idx=None):
        if noise_idx is None or noise_idx == -1:
            h = self.encoder(x, cur_max_token_num)
            h = self.quant_conv(h.latent_token)
            quant, emb_loss, info = self.quantize(h)
        else:
            h = self.encoder(x, cur_max_token_num)
            h = self.quant_conv(h.latent_token)
            quant_clean, emb_loss_clean, [h_noise, index_noise, index_clean] = self.quantize(h, top_k=50)
            index_noise = index_noise.reshape(h_noise.shape[0], h_noise.shape[1], -1)
            index_clean = index_clean.reshape(h_noise.shape[0], h_noise.shape[1], -1)
            noise_num = 1 if noise_idx < self.causal_num else self.group_size
            noise_end = noise_idx+noise_num

            quant_noise = torch.cat((quant_clean[:, :noise_idx], h_noise[:, noise_idx:noise_end]), dim=1).detach()

            noise_token = self.refine_conv(quant_noise)
            h = self.encoder(x, cur_max_token_num, noise_token=noise_token)
            h = self.quant_conv(h.latent_token)
            quant, emb_loss_noise, info = self.quantize(h, noise_end=noise_end)
            index_new = info[-1].reshape(h.shape[0], h.shape[1], -1)
            quant = torch.cat((quant_noise, quant[:, noise_end:]), dim=1)

            index_input = torch.cat((index_clean[:, :noise_idx], index_noise[:,noise_idx:noise_end], index_new[:, noise_end:]), dim=1)
            index_label = torch.cat((index_clean[:, :noise_end], index_new[:, noise_end:]), dim=1)

            index = torch.cat((index_clean, index_input, index_label), dim=-1).view(-1)
            info = (None, None, index)
            emb_loss = []
            if self.training:
                token_num = quant.shape[1]
                total_token_num = token_num * 2 - noise_idx - noise_num
                for l_clean, l_noise in zip(emb_loss_clean, emb_loss_noise):
                    clean_weight = token_num / total_token_num
                    noise_weight = (token_num - noise_idx - noise_num) / total_token_num
                    emb_loss.append(l_clean * clean_weight + l_noise * noise_weight)
            if self.training:
                quant = torch.cat((quant_clean, quant), dim=0)
        return quant, emb_loss, info

    def encode(self, x, cur_max_token_num=None, noise_idx=None):
        if noise_idx is None:
            noise_idx = -1
        quant, emb_loss, info = self.encode_image(x, cur_max_token_num=cur_max_token_num, noise_idx=noise_idx)

        if self.group_size > 1:
            s = average_cosine_similarity(rearrange(quant.detach(), 'b (l n) d -> (b l) n d', n = self.group_size))
        else:
            s = -100
        emb_loss += (s,)
        
        return quant, emb_loss, info

    def decode(self, quant, t=None, img_size=256):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant, t, img_size)
        return dec

    def decode_code(self, code_b, t=None, img_size=256, shape=None, channel_first=True):
        quant_b = self.quantize.get_codebook_entry(code_b, shape, channel_first)
        dec = self.decode(quant_b, t, img_size).pixel_value
        return dec

    def forward(self, input, cur_max_token_num=None, t=None, img_size=None, tgt_img=None, noise_idx=None, decoder_finetune=False):
        if cur_max_token_num is not None:
            if t is not None:
                assert t <= cur_max_token_num
            else:
                t = cur_max_token_num
        quant, diff, info = self.encode(input, cur_max_token_num, noise_idx=noise_idx)
        self.quant = quant
        dec = self.decode(quant, t, img_size)

        if self.training and (self.repa or self.global_token_learning):
            proj_loss_all = 0

            if self.repa:
                assert tgt_img is not None
                rescale_x = self.repa_preprocessor(tgt_img)
                z = self.repa_model.forward_features(rescale_x)[:, self.repa_model.num_prefix_tokens:]
                z_hat = dec.repa_hidden_states
                num_patch = z_hat.shape[1]
                z_hat = self.projection(z_hat).reshape(z_hat.shape[0], num_patch, -1)
                
                z_hat = F.normalize(z_hat, dim=-1)
                z_gt = F.normalize(z, dim=-1)
                proj_loss = mean_flat(-(z_gt * z_hat).sum(dim=-1))
                proj_loss = proj_loss.mean()
                proj_loss *= self.repa_loss_weight
                proj_loss_all += proj_loss

            if self.global_token_learning and not decoder_finetune:
                if self.repa and tgt_img.shape[2] == input.shape[2] and tgt_img.shape[3] == input.shape[3]:
                    z = z[:input.shape[0]]
                else:
                    rescale_x = self.repa_preprocessor(input)
                    z = self.repa_model.forward_features(rescale_x)[:, self.repa_model.num_prefix_tokens:]

                global_token = quant[:input.shape[0], 0]
                global_token_hat = self.global_token_proj(global_token)
                global_token_hat = F.normalize(global_token_hat, dim=-1)
                global_token_gt = z.mean(dim=1)
                global_token_gt = F.normalize(global_token_gt, dim=-1)
                global_token_proj_loss = mean_flat(-(global_token_hat * global_token_gt).sum(dim=-1))
                proj_loss_all += global_token_proj_loss.mean() * self.global_token_loss_weight
        
            diff += (proj_loss_all,)

        return dec.pixel_value, diff, info
    
    def save_config(self, path_to_save):
        json_dump(self.config.__dict__, path_to_save)


#################################################################################
#                              VQ Model Configs                                 #
#################################################################################
def VQ_8(**kwargs):
    return VQModel(ModelArgs(encoder_ch_mult=[1, 2, 2, 4], decoder_ch_mult=[1, 2, 2, 4], **kwargs))

def VQ_16(**kwargs):
    return VQModel(ModelArgs(encoder_ch_mult=[1, 1, 2, 2, 4], decoder_ch_mult=[1, 1, 2, 2, 4], **kwargs))


VQ_models = {
    'VQ-16': VQ_16, 'VQ-8': VQ_8,
    }