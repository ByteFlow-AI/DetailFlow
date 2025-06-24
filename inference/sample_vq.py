# Modified from:
#   SoftVQ-VAE: https://github.com/Hhhhhhao/continuous_tokenizer/blob/main/inference/reconstruct_vq.py
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from tqdm import tqdm
import os
from pathlib import Path
from PIL import Image
import numpy as np
import argparse
from torchvision.datasets import ImageFolder
from utils.misc import str2bool

from utils.data import center_crop_arr
from inference.load_vq import load_vq_model
from modelling.detail_token import find_closest_resolution
from utils.visualize import add_text_to_image

def main(args):
    
    
    # Setup PyTorch:
    assert torch.cuda.is_available(), "Sampling with DDP requires at least one GPU. sample.py supports CPU-only usage"
    torch.set_grad_enabled(False)

    # Setup DDP:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")
    ptdtype = {'none': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[args.mixed_precision]

    if args.image_size_eval is None:
        args.image_size_eval = args.image_size

    vq_model, config, config_yaml, res_deg = load_vq_model(args.vq_model, ema=args.vq_ema, device=device, eval_mode=True)
    cur_max_token_num = None
    if res_deg is not None:
        max_image_size = config_yaml['max_image_size']
        min_image_size = config_yaml['min_image_size']
        assert args.image_size >= min_image_size and args.image_size <= max_image_size
   
        assert args.image_size == find_closest_resolution(args.image_size, res_deg.best_res)
        cur_max_token_num = res_deg.decres2besttoken[args.image_size]
        print(f"ResolutionDegradation: size {args.image_size}, token {cur_max_token_num}, max size {max_image_size}")
        group_size = config['group_size']

    token_list = list(range(1*group_size, cur_max_token_num+group_size, group_size))
    res_list = [res_deg.token2dec_out_res[t] for t in token_list]

    print(f"Token list: {token_list}")
    print(f"Res: {res_list}")
        
    # Create folder to save samples:
    folder_name = (f"{args.dataset}-size-{args.image_size}-size-{args.image_size_eval}-Token-{cur_max_token_num}-seed-{args.global_seed}-EMA-{args.vq_ema}")
    sample_folder_dir = Path(f"{args.sample_dir}/{folder_name}")

    save_dir_list = [sample_folder_dir / f"Token-{token_list[i]}-Res-{res_list[i]}" for i in range(len(token_list))]
    save_dir_merge = sample_folder_dir / 'Merge'
    save_dir_gt = sample_folder_dir / 'gt'
    if rank == 0:
        for save_dir in save_dir_list:
            save_dir.mkdir(exist_ok=True, parents=True)
            print(f"Saving .png samples at {str(sample_folder_dir)}")
        save_dir_merge.mkdir(exist_ok=True, parents=True)
        save_dir_gt.mkdir(exist_ok=True, parents=True)
    dist.barrier()

    # Setup data:
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])

    n = args.per_proc_batch_size
    global_batch_size = n * dist.get_world_size()

    dataset = ImageFolder(args.data_path, transform=transform)
    num_step = 10
    
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=False,
        seed=args.global_seed
    )
    loader = DataLoader(
        dataset,
        batch_size=args.per_proc_batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )    

    # Figure out how many samples we need to generate on each GPU and how many iterations we need to run:
    
    loader = tqdm(loader) if rank == 0 else loader
    total = 0
    for x, _ in loader:
        
        if args.image_size_eval != args.image_size:
            rgb_gts = F.interpolate(x, size=(args.image_size_eval, args.image_size_eval), mode='bicubic')
        else:
            rgb_gts = x
        rgb_gts = (rgb_gts.permute(0, 2, 3, 1).to("cpu").numpy() + 1.0) / 2.0 # rgb_gt value is between [0, 1]
        x = x.to(device, non_blocking=True)
        with torch.no_grad():
            sample_list = []
            text_list = []
            for t, r, cur_save_dir in zip(token_list, res_list, save_dir_list):
                if t > cur_max_token_num:
                    break
                with torch.amp.autocast(device_type="cuda", dtype=ptdtype):
                    if res_deg is None:
                        dec_res = tgt_res = args.image_size
                    else:
                        dec_res = res_deg.token2dec_out_res[t]
                        tgt_res = res_deg.token2res[t]
                    if rank == 0:
                        print(f"Token {t}: {tgt_res} x {tgt_res} ")
                    text_list.append(f"Token {t}\n{tgt_res} x {tgt_res} ")
                    samples = vq_model(x, cur_max_token_num=cur_max_token_num, t=t, img_size=dec_res)[0]

                if samples.shape[-1] != args.image_size_eval:
                    samples = F.interpolate(samples, size=(args.image_size_eval, args.image_size_eval), mode='bicubic')
                
                samples = torch.clamp(127.5 * samples + 128, 0, 255).to(dtype=torch.uint8)
                samples = samples.permute(0, 2, 3, 1).to("cpu").numpy()

                for i, sample in enumerate(samples):
                    index = i * dist.get_world_size() + rank + total
                    Image.fromarray(sample).save(f"{cur_save_dir}/{index:06d}.png")

                sample_list.append(samples)
        
        x = torch.clamp(127.5 * x + 128, 0, 255).to(dtype=torch.uint8)
        x = x.permute(0, 2, 3, 1).to("cpu").numpy()

        show_img = []
        for i in range(x.shape[0]):
            index = i * dist.get_world_size() + rank + total
            gt = x[i]
            samples = [s[i] for s in sample_list] + [gt]
            samples = np.concatenate(samples, axis=1)
            show_img.append(samples)
            Image.fromarray(gt).save(f"{save_dir_gt}/{index:06d}.png")
        final = np.concatenate(show_img, axis=0)

        # x = np.split(x, x.shape[0], axis=0)
        # x = np.concatenate(x, axis=2)
        # final = np.concatenate((x, samples), axis=1)
        # final = np.squeeze(final, axis=0)
        index = rank + total
        img = Image.fromarray(final)
        img = add_text_to_image(img, text_list=text_list+['GT'])
        img = img.save(f"{str(save_dir_merge)}/{index:06d}.png")
            
        total += global_batch_size
        num_step -= 1
        if num_step == 0:
            break
        
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--dataset", type=str, choices=['imagenet', 'coco'], default='imagenet')
    parser.add_argument("--vq-model", type=str, default="SoftVQVAE/softvq-l-64")
    parser.add_argument("--vq-ema", type=str2bool, default=True, help="whether using ema vq model")
    parser.add_argument("--image-size", type=int, choices=[256, 384, 512], default=256)
    parser.add_argument("--image-size-eval", type=int, default=None)
    parser.add_argument("--sample-dir", type=str, default="reconstructions")
    parser.add_argument("--per-proc-batch-size", type=int, default=32)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--mixed-precision", type=str, default='bf16', choices=["none", "fp16", "bf16"]) 
    args = parser.parse_args()
    main(args)