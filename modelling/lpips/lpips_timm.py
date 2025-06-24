import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from timm import create_model


class Normalize(nn.Module):
    def __init__(self, mean, std, device=None):
        super(Normalize, self).__init__()
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mean = torch.tensor(mean).view(1, -1, 1, 1).to(device)
        self.std = torch.tensor(std).view(1, -1, 1, 1).to(device)

    def forward(self, x):
        return (x - self.mean) / self.std

class Denormalize(nn.Module):
    def __init__(self, mean, std, device=None):
        super(Denormalize, self).__init__()
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mean = torch.tensor(mean).view(1, -1, 1, 1).to(device)
        self.std = torch.tensor(std).view(1, -1, 1, 1).to(device)

    def forward(self, x):
        return x * self.std + self.mean

class NetLinLayer(nn.Module):
    """ A single linear layer which does a 1x1 conv """
    def __init__(self, chn_in, chn_out=1, use_dropout=False, use_linear=False):
        super(NetLinLayer, self).__init__()
        layers = [nn.Dropout(), ] if (use_dropout) else []
        if use_linear:
            layers += [nn.Linear(chn_in, chn_out, bias=False), ]
        else:
            layers += [nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False), ]
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)


def normalize_tensor(x,eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(x**2,dim=1,keepdim=True))
    return x/(norm_factor+eps)


def spatial_average(x, dims=[2,3], keepdim=True):
    return x.mean(dims,keepdim=keepdim)


class LPIPSTimm(nn.Module):
    def __init__(self, model_name='resnet50', 
                       intermediate_loss=False, logit_loss=True, 
                       resize=False, use_dropout=True, eval=False, 
                       dino_variants='depth12'):
        super(LPIPSTimm, self).__init__()
        
        if 'vit' in model_name:
            self.model = create_model(model_name, pretrained=True, img_size=256, patch_size=16)
        else:
            self.model = create_model(model_name, pretrained=True)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        
        self.resize = resize
        self.intermediate_loss = intermediate_loss
        self.logit_loss = logit_loss
        
        self.de_scale = Denormalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        self.scale = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        

        
        if 'resnet50' in model_name:
            if self.intermediate_loss:
                self.lin0 = NetLinLayer(2048, use_dropout=use_dropout)
                self.lin1 = NetLinLayer(1024, use_dropout=use_dropout)
                self.lin2 = NetLinLayer(512, use_dropout=use_dropout)
                self.lin3 = NetLinLayer(256, use_dropout=use_dropout)
                self.lin4 = NetLinLayer(64, use_dropout=use_dropout)
                self.lins = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4][::-1]
                self.key_depths = None 
            self.image_size = 224
        elif model_name == 'vit_small_patch14_dinov2.lvd142m' or 'vit_small_patch14_reg4_dinov2.lvd142m' or 'vit_base_patch14_reg4_dinov2.lvd142m':
            if self.intermediate_loss:    
                if dino_variants == 'depth12':
                    self.key_depths = np.arange(0, 12)
                elif dino_variants == 'depth6':
                    self.key_depths = [1, 3, 5, 7, 9, 11]
                elif dino_variants == 'depth4':
                    self.key_depths = [2, 5, 8, 11]
                print(self.key_depths)
                
                if dino_variants == 'depth12_no_train':
                    self.key_depths = np.arange(0, 12)
                    self.lins = nn.ModuleList([nn.Identity() for _ in range(12)])
                else:
                    self.lins = nn.ModuleList([NetLinLayer(self.model.embed_dim, use_dropout=use_dropout) for _ in self.key_depths])
            self.image_size = 256

            
        if eval:            
            # load pretrained model
            if model_name  == 'vit_small_patch14_dinov2.lvd142m':
                if dino_variants == 'depth12':
                    state_dict = torch.load('tokenizer/cache/dino2s_i256p16_layers12/latest.pth', map_location='cpu')
                elif dino_variants == 'depth4':
                    state_dict = torch.load('tokenizer/cache/dino2s_i256p16_layers4/latest.pth', map_location='cpu')
                elif dino_variants == 'depth6':
                    state_dict = torch.load('tokenizer/cache/dino2s_i256p16_layers6/latest.pth', map_location='cpu')
            elif model_name == 'vit_base_patch14_reg4_dinov2.lvd142m':
                if dino_variants == 'depth12':
                    state_dict = torch.load('checkpoints/dino2b_i256p16_layers12/latest.pth', map_location='cpu')
            keys = self.load_state_dict(self.get_lin_state_dict(state_dict), strict=False)
            
            for param in self.parameters():
                param.requires_grad = False
        
    
    def get_lin_state_dict(self, state_dict):
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                k = k[7:]
            
            if k.startswith('_orig_mod.'):
                k = k[10:]
            
            if not k.startswith('lins'):
                continue
                
            new_state_dict[k] = v
        print(new_state_dict['lins.0.model.1.weight'])
        return new_state_dict
            
            
    def forward(self, input, target):

        # re-normalize input and target from [-1, 1] to [0, 1]
        input = self.de_scale(input)
        target = self.de_scale(target)

        # resize input and target to 224x224
        if self.resize and input.size(-1) != self.image_size:
            input = F.interpolate(input, size=(self.image_size, self.image_size), mode='bicubic', align_corners=False, antialias=True)
            target = F.interpolate(target, size=(self.image_size, self.image_size), mode='bicubic', align_corners=False, antialias=True)
        
        # normalize to pretrained scale
        input = self.scale(input)
        target = self.scale(target)
        
        # forward pass through model
        outs0, outs0_feats = self.model.forward_intermediates(input)
        outs1, outs1_feats = self.model.forward_intermediates(target)
        logits0 = self.model.forward_head(outs0)
        logits1 = self.model.forward_head(outs1)
        
        # logits
        diffs = {}
        diff_cnt = 0
        if self.logit_loss:
            diff = (logits0 - logits1) ** 2
            diffs[diff_cnt] = diff.mean(dim=1, keepdim=True)
            diff_cnt += 1
        
        if self.intermediate_loss:
            if self.key_depths is not None:
                for i in self.key_depths:
                    diff = (normalize_tensor(outs0_feats[i]) - normalize_tensor(outs1_feats[i])) ** 2
                    diffs[diff_cnt] = diff
                    diff_cnt += 1
            else:
                for i in range(len(outs0_feats)):
                    diff = (normalize_tensor(outs0_feats[i]) - normalize_tensor(outs1_feats[i])) ** 2
                    diffs[diff_cnt] = diff
                    diff_cnt += 1
        
        # comput loss
        loss = 0.0
        diff_cnt = 0
        if self.logit_loss:
            loss += diffs[diff_cnt]
            diff_cnt += 1

        
        if self.intermediate_loss:
            
            if self.key_depths is not None:
                for i, d in enumerate(self.key_depths):
                    loss += spatial_average(self.lins[i](diffs[diff_cnt]), keepdim=True)
                    # loss += spatial_average(diffs[diff_cnt], keepdim=False)
                    diff_cnt += 1
            else:
                for i in range(len(outs0_feats)):
                    loss += spatial_average(self.lins[i](diffs[diff_cnt]), keepdim=True)
                    # loss += spatial_average(diffs[diff_cnt], keepdim=False)
                    diff_cnt += 1
                
        return loss


if __name__ == '__main__':
    model = LPIPSTimm('vit_small_patch14_dinov2.lvd142m', intermediate_loss=True, resize=True, logit_loss=False, eval=True, dino_variants='depth4')
    model.eval()
    input = torch.randn(1, 3, 256, 256)
    target = torch.randn(1, 3, 256, 256)
    loss = model(input, target)
    print(loss)