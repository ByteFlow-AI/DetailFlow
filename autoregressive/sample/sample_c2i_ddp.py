# Modified from:
#   DiT:  https://github.com/facebookresearch/DiT/blob/main/sample_ddp.py
#   LlamaGen: https://github.com/FoundationVision/LlamaGen/blob/main/autoregressive/sample/sample_c2i_ddp.py
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.nn.functional as F
import torch.distributed as dist

from tqdm import tqdm
import os
from PIL import Image
import numpy as np
from pathlib import Path
import math
import argparse

from inference.load_vq import load_vq_model
from autoregressive.sample.load_ar_model import load_ar_model
from utils.misc import str2bool
from autoregressive.models.generate import generate_group_par
from modelling.detail_token import find_closest_resolution
from utils.timer import Timer


os.environ['TORCHDYNAMO_CACHE_SIZE_LIMIT'] = '800000'


def export_images_to_npz(image_folder, total_images=50000):
    """
    Converts a folder of PNG images into a single .npz file.
    """
    image_array = []
    for idx in tqdm(range(total_images), desc="Converting images to .npz"):
        img_path = f"{image_folder}/{idx:06d}.png"
        with Image.open(img_path) as img:
            img_np = np.array(img, dtype=np.uint8)
            image_array.append(img_np)

    stacked_images = np.stack(image_array)
    assert stacked_images.shape == (total_images, stacked_images.shape[1], stacked_images.shape[2], 3)

    output_path = f"{image_folder}.npz"
    np.savez(output_path, arr_0=stacked_images)
    print(f".npz file saved at {output_path} with shape {stacked_images.shape}.")
    return output_path


def main(args):
    # Setup PyTorch:
    if not torch.cuda.is_available():
        raise RuntimeError("This script requires at least one GPU (DDP). For CPU-only usage, refer to sample.py.")
    
    # Disable gradient calculation for inference
    torch.set_grad_enabled(False)

    # Initialize the Distributed Data Parallel process group
    dist.init_process_group(backend="nccl")

    local_rank = dist.get_rank()
    total_gpus = torch.cuda.device_count()
    assigned_device = local_rank % total_gpus

    # Compute a unique seed per process for deterministic behavior
    unique_seed = args.global_seed * dist.get_world_size() + local_rank
    torch.manual_seed(unique_seed)

    # Set the current process's CUDA device
    torch.cuda.set_device(assigned_device)

    # Logging the startup information
    print(f"[DDP] Initialized process with rank={local_rank}, seed={unique_seed}, world size={dist.get_world_size()}")
    print("Parsed arguments:", args)

    sample_folder_dir = Path(args.sample_output_dir) 
    if args.eval_image_size is None:
        args.eval_image_size = args.image_size

    precision = {'none': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[args.precision]

    # create and load model
    vq_model, config, config_yaml, res_deg = load_vq_model(args.vq_checkpoint, ema=args.use_vq_ema, device=assigned_device)
    print(f"image tokenizer is loaded. EMA {args.use_vq_ema}")

    cur_max_token_num = None
    if res_deg is not None:
        max_image_size = config_yaml['max_image_size']
        min_image_size = config_yaml['min_image_size']
        assert args.image_size >= min_image_size and args.image_size <= max_image_size
        assert args.image_size == find_closest_resolution(args.image_size, res_deg.best_res)
        cur_max_token_num = res_deg.decres2besttoken[args.image_size]
    
    gpt_model, ar_config, _ = load_ar_model(args.gpt_checkpoint, ema=args.use_gpt_ema, device=assigned_device, dtype=precision, from_fsdp=args.load_from_fsdp)
    print(f"AR Model is loaded. EMA {args.use_gpt_ema}")
    block_size = ar_config['block_size']
    gpt_model.freqs_cis = gpt_model.freqs_cis.to(device=assigned_device)
    gpt_model.group_causal_mask = gpt_model.group_causal_mask.to(device=assigned_device)

    if args.enable_compile:
        print(f"compiling the model...")
        gpt_model = torch.compile(
            gpt_model,
            mode="reduce-overhead",
            fullgraph=True
        ) # requires PyTorch 2.0 (optional)
    else:
        print(f"no model compile") 

    if local_rank == 0:
        sample_folder_dir.mkdir(exist_ok=True, parents=True)
        print(f"Saving .png samples at {str(sample_folder_dir)}")

    dist.barrier()

    n = args.per_process_batch_size
    global_batch_size = n * dist.get_world_size()
    
    total_samples = int(math.ceil(args.num_fid_samples / global_batch_size) * global_batch_size)

    samples_needed_this_gpu = int(total_samples // dist.get_world_size())
    iterations = int(samples_needed_this_gpu // n)

    assert samples_needed_this_gpu % n == 0
    assert total_samples % dist.get_world_size() == 0
    if local_rank == 0:
        print(f"sampled image number: {total_samples}")

    pbar = range(iterations)
    pbar = tqdm(pbar) if local_rank == 0 else pbar
    total = 0
    timer = Timer(warmup=100)
    for _ in pbar:
        # Sample inputs:
        c_indices = torch.randint(0, args.num_classes, (n,), device=assigned_device)

        with torch.no_grad():
            with torch.amp.autocast(device_type="cuda", dtype=precision):  
                with timer:
                    index_sample = generate_group_par(
                        gpt_model, c_indices, block_size,
                        cfg_scale=args.cfg_scale, cfg_interval=args.cfg_interval,
                        temperature=args.temperature, top_k=args.top_k,
                        top_p=args.top_p, sample_logits=True, gradual_cfg=args.gradual_cfg,
                        )
                    samples = vq_model.decode_code(index_sample, img_size=args.image_size) # output value is between [-1, 1]
        if args.eval_image_size != samples.shape[-1]:
            samples = F.interpolate(samples, size=(args.eval_image_size, args.eval_image_size), mode='bicubic')

        samples = torch.clamp(127.5 * samples + 128.0, 0, 255)
        samples = samples.permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
        
        # Loop through the local batch of image samples
        for idx, img_array in enumerate(samples):
            # Calculate the global index of the current sample across all distributed processes
            global_idx = idx * dist.get_world_size() + local_rank + total

            # Convert the NumPy array to a PIL image and save it with zero-padded filename
            Image.fromarray(img_array).save(f"{sample_folder_dir}/{global_idx:06d}.png")
        total += global_batch_size

    
    timer.report()
    dist.barrier()
    if local_rank == 0:
        export_images_to_npz(sample_folder_dir, total_images=args.num_fid_samples)
        print("Processing completed.")

    # Ensure all processes reach this point before final cleanup
    dist.barrier()
    # Properly shut down the distributed training environment
    dist.destroy_process_group()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Image generation config")

    parser.add_argument("--gpt_checkpoint", type=str, default=None, help="Path to GPT checkpoint")
    parser.add_argument("--use_gpt_ema", type=str2bool, default=True, help="Use EMA for GPT model")
    parser.add_argument("--vq_checkpoint", type=str, default=None, help="Path to VQ model checkpoint")
    parser.add_argument("--use_vq_ema", type=str2bool, default=False, help="Use EMA for VQ model")
    parser.add_argument("--load_from_fsdp", action='store_true', help="Load model from FSDP format")

    parser.add_argument("--precision", type=str, choices=["none", "fp16", "bf16"], default="bf16", help="Training precision")
    parser.add_argument("--enable_compile", action='store_true', default=True, help="Enable torch.compile")

    parser.add_argument("--image_size", type=int, choices=[256, 384, 512], default=256, help="Training image size")
    parser.add_argument("--eval_image_size", type=int, default=None, help="Evaluation image size")
    parser.add_argument("--num_classes", type=int, default=1000, help="Number of output classes")

    parser.add_argument("--cfg_scale", type=float, default=1.5, help="CFG (classifier-free guidance) scale")
    parser.add_argument("--cfg_interval", type=float, default=-1, help="Interval to apply CFG")
    parser.add_argument("--gradual_cfg", type=str2bool, default=False, help="Use gradual CFG scaling")
    parser.add_argument("--top_k", type=int, default=0, help="Top-k sampling")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top-p (nucleus) sampling")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")

    parser.add_argument("--sample_output_dir", type=str, default="samples", help="Directory to save samples")
    parser.add_argument("--per_process_batch_size", type=int, default=32, help="Batch size per process")
    parser.add_argument("--num_fid_samples", type=int, default=50000, help="Number of samples for FID evaluation")
    parser.add_argument("--global_seed", type=int, default=0, help="Global random seed")

    args = parser.parse_args()
    main(args)