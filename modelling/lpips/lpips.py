import torch
import torch.nn as nn
import os
import hashlib
import requests
from tqdm import tqdm
from torchvision import models
from collections import namedtuple

# URL, file, and hash configuration
URLS = {
    "vgg_lpips": "https://heibox.uni-heidelberg.de/f/607503859c864bc1b30b/?dl=1"
}
FILENAMES = {
    "vgg_lpips": "vgg.pth"
}
HASHES = {
    "vgg_lpips": "d507d7349b931f0638a25a48a722f98a"
}

def fetch_checkpoint(url, target_path, chunk=1024):
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    with requests.get(url, stream=True) as r:
        total = int(r.headers.get('content-length', 0))
        with tqdm(total=total, unit='B', unit_scale=True) as bar:
            with open(target_path, 'wb') as f:
                for chunk_data in r.iter_content(chunk):
                    f.write(chunk_data)
                    bar.update(len(chunk_data))

def verify_md5(path):
    with open(path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

def resolve_ckpt(name, cache_dir, validate=False):
    assert name in URLS
    full_path = os.path.join(cache_dir, FILENAMES[name])
    if not os.path.exists(full_path) or (validate and verify_md5(full_path) != HASHES[name]):
        print(f"Fetching model {name} from {URLS[name]}")
        fetch_checkpoint(URLS[name], full_path)
        digest = verify_md5(full_path)
        assert digest == HASHES[name], digest
    return full_path

class LPIPS(nn.Module):
    def __init__(self, use_dropout=True):
        super().__init__()
        self.scaling_layer = ScalingLayer()
        self.chns = [64, 128, 256, 512, 512]
        self.net = vgg16(pretrained=True, requires_grad=False)
        self.lin0 = NetLinLayer(self.chns[0], use_dropout)
        self.lin1 = NetLinLayer(self.chns[1], use_dropout)
        self.lin2 = NetLinLayer(self.chns[2], use_dropout)
        self.lin3 = NetLinLayer(self.chns[3], use_dropout)
        self.lin4 = NetLinLayer(self.chns[4], use_dropout)
        self.load_from_pretrained()
        for param in self.parameters():
            param.requires_grad = False

    def load_from_pretrained(self, name="vgg_lpips"):
        ckpt_path = resolve_ckpt(name, os.getenv("CACHE_DATA_PATH", os.path.join(os.path.dirname(__file__), "cache")))
        self.load_state_dict(torch.load(ckpt_path, map_location="cpu"), strict=False)
        print(f"Pretrained LPIPS model loaded from: {ckpt_path}")

    @classmethod
    def from_pretrained(cls, name="vgg_lpips"):
        if name != "vgg_lpips":
            raise ValueError("Only vgg_lpips is supported")
        model = cls()
        ckpt_path = resolve_ckpt(name, os.path.join(os.path.dirname(__file__), "cache"))
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu"), strict=False)
        return model

    def forward(self, x1, x2):
        x1_proc, x2_proc = self.scaling_layer(x1), self.scaling_layer(x2)
        features1 = self.net(x1_proc)
        features2 = self.net(x2_proc)
        layers = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]
        loss = 0.0
        for i in range(len(self.chns)):
            f1 = normalize_tensor(features1[i])
            f2 = normalize_tensor(features2[i])
            diff = (f1 - f2) ** 2
            loss += spatial_average(layers[i].model(diff), keepdim=True)
        return loss


class ScalingLayer(nn.Module):
    def __init__(self):
        super().__init__()
        shift_values = torch.Tensor([-0.030, -0.088, -0.188])[None, :, None, None]
        scale_values = torch.Tensor([0.458, 0.448, 0.450])[None, :, None, None]
        self.register_buffer("shift", shift_values)
        self.register_buffer("scale", scale_values)

    def forward(self, img):
        return (img - self.shift) / self.scale


class NetLinLayer(nn.Module):
    def __init__(self, chn_in, use_dropout=False, chn_out=1):
        super().__init__()
        ops = [nn.Dropout()] if use_dropout else []
        ops.append(nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False))
        self.model = nn.Sequential(*ops)


class vgg16(nn.Module):
    def __init__(self, requires_grad=False, pretrained=True):
        super().__init__()
        vgg_layers = models.vgg16(pretrained=pretrained).features
        self.slice1 = nn.Sequential(*[vgg_layers[i] for i in range(4)])
        self.slice2 = nn.Sequential(*[vgg_layers[i] for i in range(4, 9)])
        self.slice3 = nn.Sequential(*[vgg_layers[i] for i in range(9, 16)])
        self.slice4 = nn.Sequential(*[vgg_layers[i] for i in range(16, 23)])
        self.slice5 = nn.Sequential(*[vgg_layers[i] for i in range(23, 30)])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        out1 = self.slice1(x)
        out2 = self.slice2(out1)
        out3 = self.slice3(out2)
        out4 = self.slice4(out3)
        out5 = self.slice5(out4)
        return namedtuple("VGGFeatures", ["relu1_2", "relu2_2", "relu3_3", "relu4_3", "relu5_3"])(out1, out2, out3, out4, out5)


def normalize_tensor(tensor, eps=1e-10):
    norm = torch.sqrt(torch.sum(tensor ** 2, dim=1, keepdim=True))
    return tensor / (norm + eps)

def spatial_average(tensor, keepdim=True):
    return tensor.mean([2, 3], keepdim=keepdim)
