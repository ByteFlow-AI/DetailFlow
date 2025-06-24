# Modified from:
#   fast-DiT: https://github.com/chuanyangjin/fast-DiT/blob/main/train.py
#   nanoGPT: https://github.com/karpathy/nanoGPT/blob/master/model.py
#   LlamaGen: https://github.com/FoundationVision/LlamaGen/blob/main/autoregressive/train/train_c2i.py
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data.distributed import DistributedSampler
from glob import glob
from copy import deepcopy
import os
import time
import inspect
import argparse
import ruamel.yaml as yaml
import wandb
from pathlib import Path

from utils.logger_func import create_logger
from utils.distributed import init_distributed_mode
from utils.ema import update_ema, requires_grad
from utils.misc import manage_checkpoints, str2bool, custom_load_model_state_dict
from torchvision.datasets import ImageFolder
from dataset.build import build_dataset
from autoregressive.models.gpt import GPT_models
from utils.data import random_crop_arr
from tools import list_files, sort_filenames, ETATimer, json_load, get_file_from_folder
from inference.load_vq import load_vq_model
import math
from torch.optim.lr_scheduler import LambdaLR

#################################################################################
#                             Training Helper Functions                         #
#################################################################################
def creat_optimizer(model, weight_decay, learning_rate, betas, logger):
    # start with all of the candidate parameters
    param_dict = {pn: p for pn, p in model.named_parameters()}
    # filter out those that do not require grad
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
    # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    logger.info(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
    logger.info(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
    # Create AdamW optimizer and use the fused version if it is available
    fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    extra_args = dict(fused=True) if fused_available else dict()
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
    logger.info(f"using fused AdamW: {fused_available}")
    return optimizer


def generate_z_input_target(z_indices, group_size, p):
    if z_indices.shape[-1] == 1:
        return z_indices[:, :-group_size, 0], z_indices[:, :, 0]
    
    assert z_indices.shape[-1] == 3
    batch_size = z_indices.size(0)
    
    noise_mask = torch.rand(batch_size, device=z_indices.device) < p
    
    z_input_no_noise = z_indices[:, :-group_size, 0]
    z_target_no_noise = z_indices[:, :, 0]
    
    z_input_with_noise = z_indices[:, :-group_size, 1]
    z_target_with_noise = z_indices[:, :, 2]
    
    z_input = torch.where(noise_mask.view(-1, 1), z_input_with_noise, z_input_no_noise)
    z_target = torch.where(noise_mask.view(-1, 1), z_target_with_noise, z_target_no_noise)
    
    return z_input, z_target


#################################################################################
#                                  Training Loop                                #
#################################################################################
def main(args):
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    
    # Setup DDP:
    init_distributed_mode(args)
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)

    # Setup an experiment folder:
    checkpoint_dir = f"{args.results_dir}/checkpoints"  # Stores saved model checkpoints
    checkpoint_dir_milestone = f"{args.results_dir}/checkpoints_milestone"
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(checkpoint_dir_milestone, exist_ok=True)
    wandb_logger = None
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.gpt_model.replace("/", "-")  # e.g., GPT-XL/2 --> GPT-XL-2 (for naming folders)
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"  # Create an experiment folder
        
        os.makedirs(experiment_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")

        experiment_config = vars(args)
        with open(os.path.join(args.results_dir, 'config.yaml'), 'w', encoding='utf-8') as f:
            # Use the round_trip_dump method to preserve the order and style
            file_yaml = yaml.YAML()
            file_yaml.dump(experiment_config, f)
        
        if not args.debug_mode:
            wandb_logger = wandb.init(project='cloud_detail_flow', name=f'exp{experiment_index:03d}-{model_string_name}-{Path(args.results_dir).name}')
    else:
        logger = create_logger(None)

    # training args
    logger.info(f"{args}")

    # training env
    logger.info(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    if args.vq_path is not None and len(args.vq_path) > 0 and args.code_path is None:
        vq_model, vq_config, _ = load_vq_model(args.vq_path, ema=args.vq_ema, device='cpu', eval_mode=True, logger=logger)
            
    if args.code_path is not None:
        config_file_list = get_file_from_folder(args.code_path, ".json")
        assert len(config_file_list) == 1
        vq_config = json_load(config_file_list[0])
        logger.info(f"Load VQ code from {config_file_list[0]}")

    image_size = vq_config['ar_image_size']
    codebook_size = vq_config['codebook_size']
    codebook_embed_dim = vq_config['codebook_embed_dim']
    num_latent_tokens = vq_config['ar_token_num']
    group_size = vq_config['group_size']
    enc_patch_size = vq_config['enc_patch_size']
    causal_num = vq_config['causal_num']
    
    # Setup model
    if args.drop_path_rate > 0.0:
        dropout_p = 0.0
    else:
        dropout_p = args.dropout_p
    if args.image_size is None:
        args.image_size = image_size

    model = GPT_models[args.gpt_model](
        vocab_size=codebook_size,
        block_size=num_latent_tokens,
        num_classes=args.num_classes,
        cls_token_num=args.cls_token_num,
        model_type=args.gpt_type,
        resid_dropout_p=dropout_p,
        ffn_dropout_p=dropout_p,
        drop_path_rate=args.drop_path_rate,
        token_dropout_p=args.token_dropout_p,
        learnable_tok_embeddings=args.learnable_tok_embeddings,
        group_size=group_size,
        causal_num=causal_num,
    ).to(device)
    model.freqs_cis = model.freqs_cis.to(device=device)
    model.group_causal_mask = model.group_causal_mask.to(device=device)

    assert model.causal_num < num_latent_tokens

    if rank == 0:
        model.save_config(f"{args.results_dir}/config.json")
    logger.info(f"GPT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    if args.ema:
        ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
        requires_grad(ema, False)
        logger.info(f"EMA Parameters: {sum(p.numel() for p in ema.parameters()):,}")

    # Setup optimizer
    optimizer = creat_optimizer(model, args.weight_decay, args.lr, (args.beta1, args.beta2), logger)

    # Setup data:
    if args.code_path is None:
        transform = transforms.Compose([
            transforms.Lambda(lambda pil_image: random_crop_arr(pil_image, args.image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])
        dataset = ImageFolder(args.data_path, transform=transform)
        dataset_info = f'{args.data_path}'
    else:
        dataset = build_dataset(args)
        flip_info = 'with' if dataset.flip else 'without'
        aug_info = 10 if 'ten_crop' in dataset.feature_dir else 1
        aug_info = 2 * aug_info if dataset.aug_feature_dir is not None else aug_info
        dataset_info = f"images ({args.code_path}) {flip_info} flip augmentation and {aug_info} crop augmentation"
    
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size // dist.get_world_size()),
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )

    num_update_steps_per_epoch = len(loader)
    max_train_steps = args.epochs * num_update_steps_per_epoch
    logger.info(f"Dataset contains {len(dataset):,}, steps per epoch {num_update_steps_per_epoch}")
    logger.info(dataset_info)


    if args.warmup_steps is not None:
        warmup_steps = args.warmup_steps
    else:
        warmup_ratio = args.warmup_ratio
        warmup_steps = int(warmup_ratio * max_train_steps)

    def lr_lambda(current_step, warmup_start_ratio=0.005, cosine_end_ratio=0.01):
        if current_step < warmup_steps:
            return warmup_start_ratio + (current_step / warmup_steps) * (1 - warmup_start_ratio)
        else:
            progress = (current_step - warmup_steps) / (max_train_steps - warmup_steps)
            return cosine_end_ratio + (1 - cosine_end_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    if args.resume_from_newest_ckpt:
        ckpt_dirs = list_files([checkpoint_dir])
        ckpt_dirs = sort_filenames(ckpt_dirs)
        if len(ckpt_dirs) > 0:
            args.gpt_ckpt = ckpt_dirs[-1]
            args.finetune = False

    # Prepare models for training:
    if args.gpt_ckpt:
        checkpoint = torch.load(args.gpt_ckpt, map_location="cpu")
        model = custom_load_model_state_dict(model=model, orig_state_dict=checkpoint['model'], strict=args.model_weight_strict)

        if args.ema:
            ema = custom_load_model_state_dict(model=ema, orig_state_dict=checkpoint["ema"] if "ema" in checkpoint else checkpoint["model"], strict=args.model_weight_strict)
        
        if not args.finetune:
            optimizer.load_state_dict(checkpoint["optimizer"])
            if "scheduler" in checkpoint:
                scheduler.load_state_dict(checkpoint["scheduler"])
                logger.info(f"Loaded scheduler from {args.gpt_ckpt}")

            train_steps = checkpoint["steps"] if "steps" in checkpoint else int(args.gpt_ckpt.split('/')[-1].split('.')[0])
            start_epoch = int(train_steps / int(len(dataset) / args.global_batch_size))
            train_steps = int(start_epoch * int(len(dataset) / args.global_batch_size))
        else:
            train_steps = 0
            start_epoch = 0
        del checkpoint
        logger.info(f"Resume training from checkpoint: {args.gpt_ckpt}")
        logger.info(f"Initial state: steps={train_steps}, epochs={start_epoch}")
    else:
        train_steps = 0
        start_epoch = 0
        if args.ema:
            update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights

    if args.compile:
        logger.info("compiling the model... (may take several minutes)")
        model = torch.compile(model) # requires PyTorch 2.0        
    
    model = DDP(model.to(device), device_ids=[args.gpu])
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    if args.ema:
        ema.eval()  # EMA model should always be in eval mode

    ptdtype = {'none': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[args.mixed_precision]
    # initialize a GradScaler. If enabled=False scaler is a no-op
    scaler = torch.cuda.amp.GradScaler(enabled=(args.mixed_precision =='fp16'))
    # Variables for monitoring/logging purposes:
    log_steps = 0
    running_loss = 0
    start_time = time.time()

    eta_timer = ETATimer()

    save_steps = args.ckpt_every if args.save_epochs is None else num_update_steps_per_epoch * args.save_epochs

    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            if args.code_path is None:
                with torch.no_grad():
                    assert NotImplementedError()
                    _, _, [_, _, z_indices] = vq_model.encode(x)
            else:
                z_indices = x
            z_indices = z_indices.reshape(x.shape[0], num_latent_tokens, -1)

            z_input, z_target = generate_z_input_target(z_indices, group_size, args.correction_prob)

            c_indices = y.reshape(-1)
            assert z_indices.shape[0] == c_indices.shape[0]
            with torch.amp.autocast(device_type="cuda", dtype=ptdtype):  
                _, loss = model(cond_idx=c_indices, idx=z_input, targets=z_target)
            # backward pass, with gradient scaling if training in fp16         
            scaler.scale(loss).backward()
            if args.max_grad_norm != 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            # step the optimizer and scaler if training in fp16
            scaler.step(optimizer)
            scheduler.step()  
            scaler.update()
            # flush the gradients as soon as we can, no need for this memory anymore
            optimizer.zero_grad(set_to_none=True)
            if args.ema:
                update_ema(ema, model.module._orig_mod if args.compile else model.module)

            # Log loss values:
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time.time()
                steps_per_sec = log_steps / (end_time - start_time)
                eta_timer.update((end_time - start_time) / log_steps)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                logger.info(f"(step={train_steps:07d}/total_steps:{max_train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}, ETA(h:m:s) {eta_timer.get_remaining_time(max_train_steps-train_steps)}")

                if rank == 0 and wandb_logger is not None:
                    log_dict = {"lr": optimizer.param_groups[0]["lr"], "train_loss": avg_loss, 'epoch': epoch}
                    wandb_logger.log(log_dict,
                        step=train_steps
                    )

                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time.time()

            # Save checkpoint:
            if train_steps % save_steps == 0 and train_steps > 0:
                if rank == 0:
                    if args.compile:
                        model_weight = model.module._orig_mod.state_dict()
                    else:
                        model_weight = model.module.state_dict()  
                    checkpoint = {
                        "model": model_weight,
                        "optimizer": optimizer.state_dict(),
                        "steps": train_steps,
                        "args": args,
                        "scheduler": scheduler.state_dict(),
                    }
                    if args.ema:
                        checkpoint["ema"] = ema.state_dict()
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                    if (epoch + 1) % 50 == 0:
                        checkpoint_path = f"{checkpoint_dir_milestone}/{epoch:04d}_{train_steps:07d}.pt"
                        torch.save(checkpoint, checkpoint_path)
                        logger.info(f"Saved milestone checkpoint to {checkpoint_path}")
                    manage_checkpoints(checkpoint_dir)
                dist.barrier()

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    logger.info("Done!")
    dist.destroy_process_group()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='configs/tokenizer/cnn_llamagen_vq16.yaml', help="config file used to specify parameters")
    parser.add_argument("--code-path", type=str, default=None)

    parser.add_argument("--gpt-model", type=str, choices=list(GPT_models.keys()), default="GPT-B")
    parser.add_argument("--gpt-ckpt", type=str, default=None, help="ckpt path for resume training")
    parser.add_argument("--finetune", type=str2bool, default=False, help="finetune a pre-trained gpt model")
    parser.add_argument("--model-weight-strict", type=str2bool, default=True, help="Whether consistent model parameter loading is required")
    parser.add_argument("--gpt-type", type=str, choices=['c2i', 't2i'], default="c2i", help="class-conditional or text-conditional")
    
    parser.add_argument("--ema", type=str2bool, default=True, help="whether using ema training")
    parser.add_argument("--cls-token-num", type=int, default=1, help="max token number of condition input")
    parser.add_argument("--dropout-p", type=float, default=0.1, help="dropout_p of resid_dropout_p and ffn_dropout_p")
    parser.add_argument("--token-dropout-p", type=float, default=0.1, help="dropout_p of token_dropout_p")
    parser.add_argument("--drop-path-rate", type=float, default=0.0, help="using stochastic depth decay")
    parser.add_argument("--compile", type=str2bool, default=True)
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--data-path", type=str, default="ImageNet2012/train")
    parser.add_argument("--dataset", type=str, default='imagenet_code')

    parser.add_argument("--vq-path", type=str, default=None)
    parser.add_argument("--vq-ema", type=str2bool, default=True, help="whether using ema vq model")
    
    parser.add_argument("--image-size", type=int, choices=[256, 384, 448, 512], default=None)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-4)

    parser.add_argument("--warmup-steps", type=int, default=None, help="Number of warmup steps for learning rate. If not set, use warmup_ratio * total_steps.")
    parser.add_argument("--warmup-ratio", type=float, default=0.05, help="Ratio of total training steps for warmup. Default is 0.05 (5%%)")


    parser.add_argument("--weight-decay", type=float, default=5e-2, help="Weight decay to use")
    parser.add_argument("--beta1", type=float, default=0.9, help="beta1 parameter for the Adam optimizer")
    parser.add_argument("--beta2", type=float, default=0.95, help="beta2 parameter for the Adam optimizer")
    parser.add_argument("--max-grad-norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=24)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=5000)
    parser.add_argument("--save-epochs", type=int, default=None, help="When save-epochs is valid, ckpt-every is ignored.")

    parser.add_argument("--mixed-precision", type=str, default='bf16', choices=["none", "fp16", "bf16"]) 

    parser.add_argument("--learnable_tok_embeddings", type=str2bool, default=True)

    parser.add_argument("--resume_from_newest_ckpt", type=str2bool, default=True)
    parser.add_argument("--debug-mode", action='store_true')
    parser.add_argument("--correction-prob", default=0.3, type=float, help="self-correction prob")

    args = parser.parse_args()

    if args.config is not None:
        with open(args.config, 'r', encoding='utf-8') as f:
            file_yaml = yaml.YAML()
            config_args = file_yaml.load(f)

            yaml_keys = set(config_args.keys())
            args_keys = set(vars(args).keys())

            missing_keys = yaml_keys - args_keys

            if missing_keys:
                raise ValueError(f"Undefined keys: {missing_keys}")

            parser.set_defaults(**config_args)
    args = parser.parse_args()
    main(args)
