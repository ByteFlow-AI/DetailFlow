# Modified from:
#   taming-transformers:  https://github.com/CompVis/taming-transformers
#   muse-maskgit-pytorch: https://github.com/lucidrains/muse-maskgit-pytorch/blob/main/muse_maskgit_pytorch/vqgan_vae.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from modelling.lpips import LPIPS, LPIPSTimm
from modelling.discriminators import PatchGANDiscriminator, StyleGANDiscriminator, PatchGANMaskBitDiscriminator, DinoDiscriminator
from utils.diff_aug import DiffAugment

import torch.distributed as tdist

def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def vanilla_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.softplus(-logits_real))
    loss_fake = torch.mean(F.softplus(logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def non_saturating_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.binary_cross_entropy_with_logits(torch.ones_like(logits_real),  logits_real))
    loss_fake = torch.mean(F.binary_cross_entropy_with_logits(torch.zeros_like(logits_fake), logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def hinge_gen_loss(logit_fake):
    return -torch.mean(logit_fake)


def non_saturating_gen_loss(logit_fake):
    return torch.mean(F.binary_cross_entropy_with_logits(torch.ones_like(logit_fake),  logit_fake))


def adopt_weight(weight, global_step, threshold=0, value=0.):
    if global_step < threshold:
        weight = value
    return weight


class LeCAM_EMA(object):
    def __init__(self, init=0., decay=0.999):
        self.logits_real_ema = init
        self.logits_fake_ema = init
        self.decay = decay

    def update(self, logits_real, logits_fake):
        self.logits_real_ema = self.logits_real_ema * self.decay + torch.mean(logits_real).item() * (1 - self.decay)
        self.logits_fake_ema = self.logits_fake_ema * self.decay + torch.mean(logits_fake).item() * (1 - self.decay)


def lecam_reg(real_pred, fake_pred, lecam_ema):
    reg = torch.mean(F.relu(real_pred - lecam_ema.logits_fake_ema).pow(2)) + \
          torch.mean(F.relu(lecam_ema.logits_real_ema - fake_pred).pow(2))
    return reg


class VQLoss(nn.Module):
    def __init__(self, disc_start, disc_loss="hinge", disc_dim=64, disc_type='patchgan', image_size=256,
                 disc_num_layers=3, disc_in_channels=3, disc_weight=1.0, disc_adaptive_weight = False,
                 gen_adv_loss='hinge', 
                 reconstruction_loss='l2', reconstruction_weight=1.0, codebook_weight=1.0, 
                 perceptual_loss='vgg', perceptual_weight=1.0, perceptual_model='vgg', perceptual_intermediate_loss=False, perceptural_logit_loss=False, perceptual_resize=False, perceptual_dino_variants='depth12_no_train', perceptual_warmup=None,
                 lecam_loss_weight=None,
                 disc_cr_loss_weight=0.0,
                 use_diff_aug=False,
                 wandb_logger=None
    ):
        super().__init__()
        # discriminator loss
        assert disc_type in ["patchgan", "stylegan", "maskbit", "dino"]
        assert disc_loss in ["hinge", "vanilla", "non-saturating"]
        if disc_type == "patchgan":
            self.discriminator = PatchGANDiscriminator(
                input_nc=disc_in_channels, 
                n_layers=disc_num_layers,
                ndf=disc_dim,
            )
        elif disc_type == "stylegan":
            self.discriminator = StyleGANDiscriminator(
                input_nc=disc_in_channels, 
                image_size=image_size,
            )
        elif disc_type == "maskbit":
            self.discriminator = PatchGANMaskBitDiscriminator(
                input_nc=disc_in_channels, 
                n_layers=disc_num_layers,
                ndf=disc_dim,
            )
        elif disc_type == "dino":
            self.discriminator = DinoDiscriminator()
        else:
            raise ValueError(f"Unknown GAN discriminator type '{disc_type}'.")
        if disc_loss == "hinge":
            self.disc_loss = hinge_d_loss
        elif disc_loss == "vanilla":
            self.disc_loss = vanilla_d_loss
        elif disc_loss == "non-saturating":
            self.disc_loss = non_saturating_d_loss
        else:
            raise ValueError(f"Unknown GAN discriminator loss '{disc_loss}'.")
        self.discriminator_iter_start = disc_start
        self.disc_weight = disc_weight
        self.disc_adaptive_weight = disc_adaptive_weight

        assert gen_adv_loss in ["hinge", "non-saturating"]
        # gen_adv_loss
        if gen_adv_loss == "hinge":
            self.gen_adv_loss = hinge_gen_loss
        elif gen_adv_loss == "non-saturating":
            self.gen_adv_loss = non_saturating_gen_loss
        else:
            raise ValueError(f"Unknown GAN generator loss '{gen_adv_loss}'.")

        # perceptual loss
        if perceptual_loss == "vgg":
            self.perceptual_loss = LPIPS().eval()
        elif perceptual_loss == "timm":
            self.perceptual_loss = LPIPSTimm(perceptual_model, perceptual_intermediate_loss, perceptural_logit_loss, perceptual_resize, eval=True, dino_variants=perceptual_dino_variants).eval()
        self.perceptual_weight = perceptual_weight
        self.perceptual_warmup = perceptual_warmup
        
        # reconstruction loss
        if reconstruction_loss == "l1":
            self.rec_loss = F.l1_loss
        elif reconstruction_loss == "l2":
            self.rec_loss = F.mse_loss
        else:
            raise ValueError(f"Unknown rec loss '{reconstruction_loss}'.")
        self.rec_weight = reconstruction_weight

        # codebook loss
        self.codebook_weight = codebook_weight

        self.lecam_loss_weight = lecam_loss_weight
        if self.lecam_loss_weight is not None:
            self.lecam_ema = LeCAM_EMA()
        
        # from var
        self.use_diff_aug = use_diff_aug
        self.disc_cr_loss_weight = disc_cr_loss_weight
            
        
        self.wandb_logger = wandb_logger

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer):
        nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
        g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()

        return d_weight.detach()

    def forward(self, codebook_loss, inputs, reconstructions, optimizer_idx, global_step, last_layer=None, 
                logger=None, log_every=100):
        # generator update
        if optimizer_idx == 0:
            # reconstruction loss
            rec_loss = self.rec_loss(inputs.contiguous(), reconstructions.contiguous())

            # perceptual loss
            p_loss = self.perceptual_loss(inputs.contiguous(), reconstructions.contiguous())
            p_loss = torch.mean(p_loss)
            if self.perceptual_warmup is  None:
                perceptual_weight = self.perceptual_weight
            else:
                perceptual_weight = min(1.0, global_step / self.perceptual_warmup) * self.perceptual_weight

            # discriminator loss
            if self.use_diff_aug:
                reconstructions = DiffAugment(reconstructions.contiguous(), policy='color,translation,cutout_0.2', prob=0.5)
            logits_fake = self.discriminator(reconstructions.contiguous())
            generator_adv_loss = self.gen_adv_loss(logits_fake)
            
            if self.disc_adaptive_weight:
                null_loss = self.rec_weight * rec_loss + perceptual_weight * p_loss
                disc_adaptive_weight = self.calculate_adaptive_weight(null_loss, generator_adv_loss, last_layer=last_layer)
            else:
                disc_adaptive_weight = 1
            disc_weight = adopt_weight(self.disc_weight, global_step, threshold=self.discriminator_iter_start)
            
            loss = self.rec_weight * rec_loss + \
                perceptual_weight * p_loss + \
                disc_adaptive_weight * disc_weight * generator_adv_loss + \
                codebook_loss[0] + codebook_loss[1] + codebook_loss[2]
            
            if len(codebook_loss) > 5:
                # repa loss
                assert len(codebook_loss) == 6
                loss += codebook_loss[5]
            
            if global_step % log_every == 0:
                rec_loss = self.rec_weight * rec_loss
                p_loss = perceptual_weight * p_loss
                generator_adv_loss = disc_adaptive_weight * disc_weight * generator_adv_loss
                repa_loss = codebook_loss[5] if len(codebook_loss) > 5 else 0.0
                logger.info(f"(Generator) rec_loss: {rec_loss:.4f}, perceptual_loss: {p_loss:.4f}, "
                            f"vq_loss: {codebook_loss[0]:.4f}, commit_loss: {codebook_loss[1]:.4f}, entropy_loss: {codebook_loss[2]:.4f}, repa_loss: {repa_loss:.4f}, "
                            f"fea_cosine_sim: {codebook_loss[4]:.4f}"
                            f"codebook_usage: {codebook_loss[3]:.4f}, generator_adv_loss: {generator_adv_loss:.4f}, "
                            f"disc_adaptive_weight: {disc_adaptive_weight:.4f}, disc_weight: {disc_weight:.4f}")
                if tdist.get_rank() == 0 and self.wandb_logger is not None:
                    self.wandb_logger.log({
                        "rec_loss": rec_loss,
                        "perceptual_loss": p_loss,
                        'repa_loss': repa_loss,
                        "codebook_loss": codebook_loss[0],
                        "codebook_usage": codebook_loss[3],
                        "fea_cosine_sim": codebook_loss[4],
                        "generator_adv_loss": generator_adv_loss,
                        "disc_adaptive_weight": disc_adaptive_weight,
                        "disc_weight": disc_weight,
                    },
                    step=global_step)
            return loss

        # discriminator update
        if optimizer_idx == 1:
            
            if self.use_diff_aug:
                logits_real = self.discriminator(DiffAugment(inputs.contiguous().detach(), policy='color,translation,cutout_0.2', prob=0.5))
                logits_fake = self.discriminator(DiffAugment(reconstructions.contiguous().detach(), policy='color,translation,cutout_0.2', prob=0.5))
            else:
                logits_real = self.discriminator(inputs.contiguous().detach())
                logits_fake = self.discriminator(reconstructions.contiguous().detach())

            disc_weight = adopt_weight(self.disc_weight, global_step, threshold=self.discriminator_iter_start)

            if self.lecam_loss_weight is not None:
                self.lecam_ema.update(logits_real, logits_fake)
                lecam_loss = lecam_reg(logits_real, logits_fake, self.lecam_ema)
                non_saturate_d_loss = self.disc_loss(logits_real, logits_fake)
                d_adversarial_loss = disc_weight * (lecam_loss * self.lecam_loss_weight + non_saturate_d_loss)
            else:
                d_adversarial_loss = disc_weight * self.disc_loss(logits_real, logits_fake)

            
            if self.disc_cr_loss_weight:
                logits_real_s = self.discriminator(DiffAugment(inputs.contiguous().detach(), policy='color,translation,cutout_0.5', prob=1.0))
                logits_fake_s = self.discriminator(DiffAugment(reconstructions.contiguous().detach(), policy='color,translation,cutout_0.5', prob=1.0))
                disc_cr_loss_weight = self.disc_cr_loss_weight if global_step >= self.discriminator_iter_start else 0.0
                d_cr = F.mse_loss(torch.cat([logits_real, logits_fake], dim=0), torch.cat([logits_real_s, logits_fake_s])) * disc_cr_loss_weight
                d_adversarial_loss += d_cr

            if global_step % log_every == 0:
                logits_real = logits_real.detach()
                logits_fake = logits_fake.detach()
                log_info = {
                            "discriminator_adv_loss": d_adversarial_loss,
                            'logits_real': logits_real.mean(),
                            "logits_fake": logits_fake.mean()
                        }
                    
                if self.disc_cr_loss_weight:
                    log_info['discriminator_cr_loss'] = d_cr

                log_string = "(Discriminator) "
                for k, v in log_info.items():
                    log_string += f"{k}: {v:.4f}, "
                logger.info(log_string)
                if tdist.get_rank() == 0 and self.wandb_logger is not None:
                    self.wandb_logger.log(log_info, step=global_step)
            return d_adversarial_loss
