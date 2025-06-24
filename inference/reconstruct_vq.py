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
import itertools
from utils.misc import str2bool
from torchvision.datasets import ImageFolder

from skimage.metrics import peak_signal_noise_ratio as psnr_loss
from skimage.metrics import structural_similarity as ssim_loss

from utils.data import center_crop_arr
from torchmetrics.image.fid import FrechetInceptionDistance
from inference.load_vq import load_vq_model
from modelling.detail_token import find_closest_resolution


def create_npz_from_sample_folder(sample_dir, num=50000):
    """
    Builds a single .npz file from a folder of .png samples.
    """
    samples = []
    for i in tqdm(range(num), desc="Building .npz file from samples"):
        sample_pil = Image.open(f"{sample_dir}/{i:06d}.png")
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    samples = np.stack(samples)
    assert samples.shape == (num, samples.shape[1], samples.shape[2], 3)
    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path



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
        correction_training = config['correction_training']
        if not correction_training:
            args.noisy_eval = False

        assert args.image_size >= min_image_size and args.image_size <= max_image_size
   
        assert args.image_size == find_closest_resolution(args.image_size, res_deg.best_res)
        cur_max_token_num = res_deg.decres2besttoken[args.image_size]
        print(f"ResolutionDegradation: size {args.image_size}, token {cur_max_token_num}, max size {max_image_size}")
    
    t = cur_max_token_num
    group_num = t // res_deg.group_size

    noise_idxs = [-1] if not args.noisy_eval else [-1] + [i * res_deg.group_size for i in [0, group_num // 4, group_num // 4 * 2, group_num // 4 * 3, group_num-1]]
    print("Noise idx list ", noise_idxs)

    for noise_idx in noise_idxs:
        # Create folder to save samples:
        folder_name = (f"{args.dataset}-size-{args.image_size}-size-{args.image_size_eval}-Token-{t}-seed-{args.global_seed}-EMA-{args.vq_ema}-Noisy-{noise_idx}")
        sample_folder_dir = f"{args.sample_dir}/{folder_name}"
        if rank == 0:
            os.makedirs(sample_folder_dir, exist_ok=True)
            print(f"Saving .png samples at {sample_folder_dir}")
        dist.barrier()

        # Setup data:
        transform = transforms.Compose([
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])

        dataset = ImageFolder(args.data_path, transform=transform)
        num_fid_samples = 50000
        
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
        n = args.per_proc_batch_size
        global_batch_size = n * dist.get_world_size()
        
        psnr_val_rgb = []
        ssim_val_rgb = []
        compute_fid_score = FrechetInceptionDistance(normalize=False).cuda()
        all_z = []
        all_zq = []
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
                with torch.amp.autocast(device_type="cuda", dtype=ptdtype):  
                    samples = vq_model(x, cur_max_token_num=cur_max_token_num, t=t, img_size=args.image_size, noise_idx=noise_idx)[0]

                if args.image_size_eval != samples.shape[-1]:
                    samples = F.interpolate(samples, size=(args.image_size_eval, args.image_size_eval), mode='bicubic')
            
            
            samples = torch.clamp(127.5 * samples + 128, 0, 255).to(dtype=torch.uint8)
            x = torch.clamp(127.5 * x + 128, 0, 255).to(dtype=torch.uint8)
        
            compute_fid_score.update(x, real=True)
            compute_fid_score.update(samples, real=False)
            
            samples = samples.permute(0, 2, 3, 1).to("cpu").numpy()

            # Save samples to disk as individual .png files
            for i, (sample, rgb_gt) in enumerate(zip(samples, rgb_gts)):
                index = i * dist.get_world_size() + rank + total
                Image.fromarray(sample).save(f"{sample_folder_dir}/{index:06d}.png")
                # metric
                rgb_restored = sample.astype(np.float32) / 255. # rgb_restored value is between [0, 1]
                psnr = psnr_loss(rgb_restored, rgb_gt)
                ssim = ssim_loss(rgb_restored, rgb_gt, multichannel=True, data_range=2.0, channel_axis=-1)
                psnr_val_rgb.append(psnr)
                ssim_val_rgb.append(ssim)
                
            total += global_batch_size
            
        fid = compute_fid_score.compute().detach()
        
        # ------------------------------------
        #       Summary
        # ------------------------------------
        # Make sure all processes have finished saving their samples
        dist.barrier()
        world_size = dist.get_world_size()
        gather_psnr_val = [None for _ in range(world_size)]
        gather_ssim_val = [None for _ in range(world_size)]
        dist.all_gather_object(gather_psnr_val, psnr_val_rgb)
        dist.all_gather_object(gather_ssim_val, ssim_val_rgb)
        

        # print(gather_latents)
        if rank == 0:
            gather_psnr_val = list(itertools.chain(*gather_psnr_val))
            gather_ssim_val = list(itertools.chain(*gather_ssim_val))   
            # gather_fid_val = list(itertools.chain(*gather_fid_val))     
            psnr_val_rgb = sum(gather_psnr_val) / len(gather_psnr_val)
            ssim_val_rgb = sum(gather_ssim_val) / len(gather_ssim_val)
            # fid_val = sum(gather_fid_val) / len(gather_fid_val)
            
            print("PSNR: %f, SSIM: %f " % (psnr_val_rgb, ssim_val_rgb))
            print("FID: %f" % (fid))
            
            result_file = f"{sample_folder_dir}_results.txt"
            print("writing results to {}".format(result_file))
            with open(result_file, 'w') as f:
                print("PSNR: %f, SSIM: %f " % (psnr_val_rgb, ssim_val_rgb), file=f)
                print("FID: %f " % (fid), file=f)
                
            
            # create_npz_from_sample_folder(sample_folder_dir, num_fid_samples)
            print("Done.")

        
        dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--dataset", type=str, choices=['imagenet', 'coco'], default='imagenet')
    parser.add_argument("--vq-model", type=str, default="SoftVQVAE/softvq-l-64")
    parser.add_argument("--vq-ema", type=str2bool, default=False, help="whether using ema vq model")
    parser.add_argument("--image-size", type=int, choices=[256, 384, 512], default=256)
    parser.add_argument("--image-size-eval", type=int, default=None)
    parser.add_argument("--sample-dir", type=str, default="reconstructions")
    parser.add_argument("--per-proc-batch-size", type=int, default=32)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--noisy-eval", type=str2bool, default=True, help="evaluate the model under disturbance")
    parser.add_argument("--mixed-precision", type=str, default='bf16', choices=["none", "fp16", "bf16"]) 
    args = parser.parse_args()
    main(args)