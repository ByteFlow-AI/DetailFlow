# Modified from:
#   fast-DiT: https://github.com/chuanyangjin/fast-DiT/blob/main/extract_features.py
#   LlamaGen: https://github.com/FoundationVision/LlamaGen/blob/main/autoregressive/train/extract_codes_c2i.py
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
import numpy as np
import argparse
import os

from utils.distributed import init_distributed_mode
from utils.data import center_crop_arr
from torchvision.datasets import ImageFolder
from inference.load_vq import load_vq_model
from utils.misc import str2bool
from tqdm import tqdm
from pathlib import Path
from tools import json_dump, get_file_from_folder
from modelling.detail_token import find_closest_resolution

#################################################################################
#                                  Training Loop                                #
#################################################################################
def main(args):
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    json_path = Path(args.code_path) / f"{args.dataset}.json"
    if json_path.exists():
        return
    
    ptdtype = {'none': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[args.mixed_precision]
    
    # Setup DDP:
    if not args.debug:
        init_distributed_mode(args)
        rank = dist.get_rank()
        device = rank % torch.cuda.device_count()
        seed = args.global_seed * dist.get_world_size() + rank
        torch.manual_seed(seed)
        torch.cuda.set_device(device)
        print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")
    else:
        device = 'cuda'
        rank = 0

    # create and load model
    vq_model, vq_config, config_yaml, res_deg = load_vq_model(args.vq_path, ema=args.vq_ema, device=device, eval_mode=True)
    os.makedirs(args.code_path, exist_ok=True)

    if args.image_size is None:
        args.image_size = vq_config['image_size']

    cur_max_token_num = vq_config['num_latent_tokens']
    correction_training = False
    if res_deg is not None:
        max_image_size = config_yaml['max_image_size']
        min_image_size = config_yaml['min_image_size']
        correction_training = vq_config['correction_training']
        assert args.image_size >= min_image_size and args.image_size <= max_image_size
        assert args.image_size == find_closest_resolution(args.image_size, res_deg.best_res)
        cur_max_token_num = res_deg.decres2besttoken[args.image_size]
        print(f"ResolutionDegradation: size {args.image_size}, token {cur_max_token_num}, max size {max_image_size}")

    if rank == 0:
        vq_config['ar_image_size'] = args.image_size
        vq_config['ar_token_num'] = cur_max_token_num
        json_dump(vq_config, f'{args.code_path}/{args.dataset}.json')

    os.makedirs(os.path.join(args.code_path, f'{args.dataset}{args.image_size}_codes'), exist_ok=True)
    os.makedirs(os.path.join(args.code_path, f'{args.dataset}{args.image_size}_labels'), exist_ok=True)

    # Setup data:
    if args.ten_crop:
        crop_size = int(args.image_size * args.crop_range)
        transform = transforms.Compose([
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, crop_size)),
            transforms.TenCrop(args.image_size), # this is a tuple of PIL Images
            transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])), # returns a 4D tensor
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])
    else:
        crop_size = args.image_size 
        transform = transforms.Compose([
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, crop_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])
    dataset = ImageFolder(args.data_path, transform=transform)
    print(f"Dataset size: {len(dataset)}. AR Image Size {args.image_size}. AR Token Num {cur_max_token_num}")
    if not args.debug:
        sampler = DistributedSampler(
            dataset,
            num_replicas=dist.get_world_size(),
            rank=rank,
            shuffle=False,
            seed=args.global_seed
        )
    else:
        sampler = None
    loader = DataLoader(
        dataset,
        batch_size=1, # important!
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )

    total = 0
    for x, y in tqdm(loader, disable=rank!=0):
        x = x.to(device)
        if args.ten_crop:
            x_all = x.flatten(0, 1)
            num_aug = 10
        else:
            x_flip = torch.flip(x, dims=[-1])
            x_all = torch.cat([x, x_flip])
            num_aug = 2
        y = y.to(device)
        with torch.no_grad():
            with torch.amp.autocast(device_type="cuda", dtype=ptdtype): 
                noise_idx = res_deg.get_noise_token_idx(cur_max_token_num, 1, correction_training=correction_training)
                _, _, [_, _, indices] = vq_model.encode(x_all, cur_max_token_num=cur_max_token_num, noise_idx=noise_idx)
                codes = indices.reshape(x.shape[0], num_aug, cur_max_token_num, -1)
                
        x = codes.detach().cpu().numpy()    # (1, num_aug, args.image_size//16 * args.image_size//16)
        train_steps = rank + total
        np.save(f'{args.code_path}/{args.dataset}{args.image_size}_codes/{train_steps}.npy', x)

        y = y.detach().cpu().numpy()    # (1,)
        np.save(f'{args.code_path}/{args.dataset}{args.image_size}_labels/{train_steps}.npy', y)
        if not args.debug:
            total += dist.get_world_size()
        else:
            total += 1

    dist.barrier()
    dist.destroy_process_group()

    npy_files = get_file_from_folder(f'{args.code_path}/{args.dataset}{args.image_size}_codes', '.npy')
    # assert len(npy_files) == len(dataset), f"code num {len(npy_files)} != dataset num {len(dataset)}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--code-path", type=str, required=True)
    parser.add_argument("--dataset", type=str, default='imagenet')
    parser.add_argument("--image-size", type=int, choices=[256, 384, 448, 512], default=None)
    parser.add_argument("--ten-crop", action='store_true', help="whether using random crop")
    parser.add_argument("--crop-range", type=float, default=1.1, help="expanding range of center crop")
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=24)
    parser.add_argument("--debug", action='store_true')
    parser.add_argument("--mixed-precision", type=str, default='bf16', choices=["none", "fp16", "bf16"])

    parser.add_argument("--vq-path", type=str, default="")
    parser.add_argument("--vq-ema", type=str2bool, default=True, help="whether using ema vq model")
    
    args = parser.parse_args()
    main(args)
