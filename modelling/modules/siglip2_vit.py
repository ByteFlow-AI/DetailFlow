import torch
import math
import os
import torch.nn as nn
import torch.nn.functional as F
import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union
from transformers import Siglip2VisionModel, Siglip2VisionConfig
from transformers import AutoConfig
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask
from utils.misc import check_flash_attn_2_available
from timm.layers import trunc_normal_

from modelling.detail_token import create_attn_mask_for_x_z
from modelling.modules.utils import CustomModelOutput


def convert_image_to_patches(image: "torch.Tensor", patch_size: int) -> "torch.Tensor":
    batch_size, num_channels, image_height, image_width = image.shape
    num_patches_height = image_height // patch_size
    num_patches_width = image_width // patch_size
    patched_image = image.reshape(batch_size, num_channels, num_patches_height, patch_size, num_patches_width, patch_size)
    patched_image = patched_image.permute(0, 2, 4, 3, 5, 1)
    patched_image = patched_image.reshape(batch_size, num_patches_height * num_patches_width, -1)
    return patched_image


class ToPixel(nn.Module):
    def __init__(self, to_pixel='linear', in_channels=3, in_dim=512, patch_size=16) -> None:
        super().__init__()
        self.to_pixel_name = to_pixel
        self.patch_size = patch_size
        self.in_channels = in_channels
        if to_pixel == 'linear':
            self.model = nn.Linear(in_dim, in_channels * patch_size * patch_size)
        elif to_pixel == 'conv':
            raise NotImplementedError
        elif to_pixel == 'identity':
            self.model = nn.Identity()
        else:
            raise NotImplementedError

    def get_last_layer(self):
        if self.to_pixel_name == 'linear':
            return self.model.weight
        elif self.to_pixel_name == 'conv':
            return self.model[1].weight
        else:
            return None

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_size
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def forward(self, x):
        if self.to_pixel_name == 'linear':
            x = self.model(x)
            x = self.unpatchify(x)
        elif self.to_pixel_name == 'conv':
            x = self.model(x)
        elif self.to_pixel_name == 'identity':
            pass
        return x


class CustomSiglip2VisionModel(nn.Module):
    def __init__(self, pretrained, layer_num=None, patch_size=None, causal_mode=False, embed_dim=None, num_heads=None, **kwargs):
        super().__init__()
        logger = logging.getLogger(__name__)
        pretrain_model_dir = str(Path(os.getenv("CACHE_DATA_PATH")) / 'siglip2-so400m-patch16-naflex')
        self.causal_mode = causal_mode
        if check_flash_attn_2_available() and not causal_mode:
            attn_implementation = "flash_attention_2"
        else:
            attn_implementation = 'sdpa'
        if pretrained:
            visual_model = Siglip2VisionModel.from_pretrained(pretrain_model_dir, attn_implementation=attn_implementation).vision_model
            logger.info(f"Load pretrained siglip2 weight from {pretrain_model_dir}")
            num_hidden_layers = visual_model.config.num_hidden_layers
            if layer_num is not None:
                assert layer_num <= num_hidden_layers
                visual_model.encoder.layers = visual_model.encoder.layers[:layer_num]
        else:
            config = Siglip2VisionConfig.from_pretrained(
                pretrain_model_dir, 
                attn_implementation=attn_implementation,
                num_hidden_layers=layer_num,
                hidden_size=embed_dim,
                num_attention_heads=num_heads,
                intermediate_size=int(embed_dim * 4),
                )
            visual_model = Siglip2VisionModel(config=config).vision_model
        visual_model.head = nn.Identity()
        visual_model.use_head = False
        
        self.attn_implementation = attn_implementation
        self.visual_model = visual_model
        self.raw_patch_size = self.visual_model.config.patch_size
        self.patch_size = patch_size if patch_size is not None else self.raw_patch_size
        self.embed_dim = self.visual_model.config.hidden_size
        self.num_hidden_layers = self.visual_model.config.num_hidden_layers
        self.layer_num = layer_num
        self.num_prefix_tokens = 0
    
    def preprocess(self, images):
        B, C, H, W = images.shape
        assert H % self.patch_size == 0
        assert W % self.patch_size == 0
        images = convert_image_to_patches(images, self.patch_size)
        grid_h = H // self.patch_size
        grid_w = W // self.patch_size
        attention_mask = torch.ones((B, grid_h*grid_w), device=images.device)
        spatial_shapes = torch.tensor([grid_h, grid_w], device=images.device)
        spatial_shapes = spatial_shapes.reshape(1, -1).repeat(B, 1)
        return images, attention_mask, spatial_shapes

    def forward_features(
        self,
        pixel_values: torch.FloatTensor,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        return_dict = return_dict if return_dict is not None else self.visual_model.config.use_return_dict

        pixel_values, pixel_attention_mask, spatial_shapes = self.preprocess(images=pixel_values)

        last_hidden_state = self.visual_model(
            pixel_values=pixel_values,
            attention_mask=pixel_attention_mask,
            spatial_shapes=spatial_shapes,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )['last_hidden_state']
        last_hidden_state = self.visual_model.post_layernorm(last_hidden_state)
        return last_hidden_state
    

class CustomSiglip(CustomSiglip2VisionModel):
    def __init__(self, pretrained, num_latent_tokens, layer_num=None, causal_mode=False, patch_size=14, group_size=1, causal_num=None, **kwargs):
        super().__init__(pretrained, layer_num, patch_size, causal_mode, **kwargs)
        self.num_latent_tokens = num_latent_tokens
        self._use_flash_attention_2 = self.attn_implementation == "flash_attention_2"
        self.group_size = group_size
        self.causal_num = self.group_size if causal_num is None else causal_num
        assert self.causal_num % self.group_size == 0

        self.latent_pos_embed = nn.Parameter(torch.zeros(1, self.num_latent_tokens, self.embed_dim))
        trunc_normal_(self.latent_pos_embed, std=.02)
    
    @staticmethod
    def resize_positional_embeddings(
        positional_embeddings: torch.Tensor,
        spatial_shapes: torch.LongTensor,
        batch_size: int,
    ) -> torch.Tensor:
        embed_dim = positional_embeddings.shape[-1]
        source_dtype = positional_embeddings.dtype
        positional_embeddings = positional_embeddings.permute(2, 0, 1).unsqueeze(0)

        height, width = spatial_shapes

        resized_embeddings = F.interpolate(
            positional_embeddings,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )

        # (1, dim, target_height, target_width) -> (target_height * target_width, dim)
        resized_embeddings = resized_embeddings.reshape(1, embed_dim, height * width).transpose(1, 2)

        resized_embeddings = resized_embeddings.to(source_dtype)
        resized_embeddings = resized_embeddings.repeat(batch_size, 1, 1)
        return resized_embeddings
    
    def patch_embedding(self, pixel_values, spatial_shapes):
        embeddings =  self.visual_model.embeddings
        target_dtype = embeddings.position_embedding.weight.dtype
        patch_embeds = embeddings.patch_embedding(pixel_values.to(dtype=target_dtype))

        positional_embeddings = embeddings.position_embedding.weight.reshape(
            embeddings.position_embedding_size, embeddings.position_embedding_size, -1
        )

        resized_positional_embeddings = self.resize_positional_embeddings(
            positional_embeddings, spatial_shapes, patch_embeds.shape[0]
        )
        fea = patch_embeds + resized_positional_embeddings
        return fea
    
    def encoder_forward(self, hidden_states, attention_mask, patch_num, repa_layer=None):
        output_attentions = False
        encoder = self.visual_model.encoder
        B = hidden_states.shape[0]
        repa_z = None

        for layer_id, encoder_layer in enumerate(encoder.layers):
            if encoder.gradient_checkpointing and encoder.training:
                layer_outputs = encoder._gradient_checkpointing_func(
                    encoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    output_attentions,
                )
            else:
                layer_outputs = encoder_layer(
                    hidden_states,
                    attention_mask,
                    output_attentions=output_attentions,
                )

            hidden_states = layer_outputs[0]

            if repa_layer is not None and layer_id == repa_layer and self.training:
                repa_z = hidden_states.view(B, -1, hidden_states.shape[-1])[:, :patch_num]
        
        return hidden_states, repa_z


    def forward(self, hidden_states, z, repa_layer=None):
        B, patch_num = hidden_states.shape[:2]

        z = z + self.latent_pos_embed[:, :z.shape[1]]
        hidden_states = torch.cat((hidden_states, z), dim=1)

        if self.causal_mode:
            if self.is_encoder:
                encoder_attention_mask = create_attn_mask_for_x_z(patch_num, z.size(1), z_causal=True, x_see_z=False, z_see_x=True, group_size=self.group_size, causal_num=self.causal_num)
            else:
                encoder_attention_mask = create_attn_mask_for_x_z(patch_num, z.size(1), z_causal=True, x_see_z=True, z_see_x=False, group_size=self.group_size, causal_num=self.causal_num)
            inverted_mask = torch.zeros_like(encoder_attention_mask, dtype=hidden_states.dtype, device=hidden_states.device)
            encoder_attention_mask = inverted_mask.masked_fill(encoder_attention_mask.to(hidden_states.device), torch.finfo(hidden_states.dtype).min)
            encoder_attention_mask = encoder_attention_mask.unsqueeze(0).unsqueeze(0).expand(B, -1, -1, -1).to(hidden_states.device)
        else:
            attention_mask = torch.ones(hidden_states.shape[:2], device=hidden_states.device)
            
            if not self._use_flash_attention_2:
                # [batch_size, seq_len] -> [batch_size, 1, tgt_seq_len, src_seq_len]
                encoder_attention_mask = _prepare_4d_attention_mask(attention_mask, hidden_states.dtype)
            else:
                encoder_attention_mask = attention_mask

        last_hidden_state, repa_z = self.encoder_forward(hidden_states, encoder_attention_mask, patch_num=patch_num, repa_layer=repa_layer)
        last_hidden_state = self.visual_model.post_layernorm(last_hidden_state)
        
        return CustomModelOutput(
            hidden_states=last_hidden_state[:, :patch_num],
            repa_hidden_states=repa_z,
            latent_token=last_hidden_state[:, patch_num:]
        )
    

class SiglipEncoder(CustomSiglip):
    def __init__(self, **kwargs):
        self.is_encoder = True

        super().__init__(**kwargs)

        self.latent_tokens = nn.Parameter(torch.zeros(1, self.num_latent_tokens, self.embed_dim))
        nn.init.normal_(self.latent_tokens, std=.02)

        self.refined_tokens = nn.Parameter(torch.zeros(1, self.num_latent_tokens, self.embed_dim))
        nn.init.normal_(self.refined_tokens, std=.02)
        
        if self.patch_size != self.raw_patch_size:
            self.visual_model.embeddings.patch_size = self.patch_size
            self.visual_model.embeddings.patch_embedding = nn.Linear(
                in_features=self.visual_model.embeddings.config.num_channels * self.patch_size * self.patch_size,
                out_features=self.visual_model.embeddings.embed_dim,
            )

    def no_weight_decay(self):
        if self.patch_size == self.raw_patch_size:
            return ['latent_tokens', 'latent_pos_embed']
        else:
            return ['latent_tokens', 'visual_model.embeddings.patch_embed', 'latent_pos_embed']
    
    def preprocess(self, images):
        B, C, H, W = images.shape
        assert H % self.patch_size == 0
        assert W % self.patch_size == 0
        images = convert_image_to_patches(images, self.patch_size)
        grid_h = H // self.patch_size
        grid_w = W // self.patch_size
        # attention_mask = torch.ones((B, grid_h*grid_w + cur_token_num), device=images.device)
        # spatial_shapes = torch.tensor([grid_h, grid_w], device=images.device)
        # spatial_shapes = spatial_shapes.reshape(1, -1).repeat(B, 1)
        return images, (grid_h, grid_w)
    
    def forward(self, x, cur_max_token_num=None, noise_token=None):
        if cur_max_token_num is None:
            cur_max_token_num = self.num_latent_tokens
        assert cur_max_token_num <= self.num_latent_tokens

        flatten_patches, spatial_shapes = self.preprocess(x)

        hidden_states = self.patch_embedding(flatten_patches, spatial_shapes)

        if noise_token is None:
            z = self.latent_tokens[:, :cur_max_token_num].expand(flatten_patches.shape[0], -1, -1)
        else:
            refined_tokens = self.refined_tokens[:, noise_token.shape[1]:cur_max_token_num].expand(flatten_patches.shape[0], -1, -1)
            z = torch.cat((noise_token, refined_tokens), dim=1)
        out = super().forward(hidden_states, z)
        
        return out
    

class SiglipDecoder(CustomSiglip):
    def __init__(self, in_channels=3, to_pixel='linear', repa_layer=None, **kwargs):
        self.is_encoder = False
        super().__init__(**kwargs)
        self.repa_layer = repa_layer

        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        nn.init.normal_(self.mask_token, std=.02)
        self.to_pixel = ToPixel(to_pixel=to_pixel, in_channels=in_channels,
                                in_dim=self.embed_dim, patch_size=self.patch_size)
        self.visual_model.embeddings.patch_embedding = nn.Identity()
    
    def no_weight_decay(self):
        return ['mask_token', 'latent_pos_embed']

    @property
    def last_layer(self):
        return self.to_pixel.model.weight
    
    def forward(self, z, t=None, img_size=256):
        if t is None:
            t = z.shape[1]

        B, _, _ = z.shape
        
        assert img_size % self.patch_size == 0
        grid_h = img_size // self.patch_size
        
        hidden_states = self.mask_token.expand(z.size(0), grid_h*grid_h, -1)

        hidden_states = self.patch_embedding(hidden_states, (grid_h, grid_h))

        out = super().forward(hidden_states, z[:, :t], repa_layer=self.repa_layer)

        pixel_value = self.to_pixel(out.hidden_states)
        return CustomModelOutput(
            hidden_states=out.hidden_states,
            repa_hidden_states=out.repa_hidden_states,
            latent_token=out.latent_token,
            pixel_value=pixel_value
        )


if __name__ == '__main__':
    imgs = torch.randint(0, 255, (2, 3, 256, 256), device='cuda')

    model = CustomSiglip2VisionModel(pretrained=False).eval().to(imgs.device)
    fea = model(pixel_values=imgs, output_attentions=False, output_hidden_states=False, return_dict=True)[0]
    print(fea.shape)