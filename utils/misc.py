import torch
import os
import numpy as np
import random

import datetime
import functools
import glob
import os
import subprocess
import sys
import time
import logging
from collections import defaultdict, deque
from typing import Iterator, List, Tuple

import numpy as np
import torch
import torch.distributed as tdist
from transformers.utils import is_flash_attn_2_available

import argparse


def check_flash_attn_2_available():
    available_flag = True
    if not is_flash_attn_2_available():
        available_flag = False
    unavailable_device = ['V100', 'Ascend']
    for d in unavailable_device:
        if d in os.getenv("ARNOLD_DEVICE_TYPE", "A800"):
            available_flag = False
    return available_flag


def str2bool(v):
    """
    str to bool
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
    

def manage_checkpoints(save_dir, keep_last_n=10):
    # List all checkpoint files in the save directory
    checkpoints = [f for f in os.listdir(save_dir) if f.endswith('.pt')]
    checkpoints = [f for f in checkpoints if 'best_ckpt' not in f]
    checkpoints.sort(key=lambda f: int(f.split('/')[-1].split('.')[0]))  # Sort by epoch number

    # If more than `keep_last_n` checkpoints exist, remove the oldest ones
    if len(checkpoints) > keep_last_n + 1:  # keep_last_n + 1 to account for the latest checkpoint
        for checkpoint_file in checkpoints[:-keep_last_n-1]:
            checkpoint_path = os.path.join(save_dir, checkpoint_file)
            if os.path.exists(checkpoint_path):
                os.remove(checkpoint_path)
                print(f"Removed old checkpoint: {checkpoint_path}")


def load_model_state_dict(orig_state_dict):
    model_state = {}
    for key, value in orig_state_dict.items():
        if key.startswith("module."):
            model_state[key[7:]] = value
        if key.startswith("_orig_mod."):
            model_state[key[10:]] = value
        else:
            model_state[key] = value
    return model_state


def custom_load_model_state_dict(model, orig_state_dict, strict):
    if strict:
        model.load_state_dict(load_model_state_dict(orig_state_dict), strict=strict)
    else:
        model = load_partial_state_dict(model, load_model_state_dict(orig_state_dict))
    return model


def load_partial_state_dict(model, pretrained_dict):
    model_dict = model.state_dict()

    matched_dict = {}
    mismatched_layers = []
    for k, v in pretrained_dict.items():
        if k in model_dict:
            if model_dict[k].shape == v.shape:
                matched_dict[k] = v
            else:
                mismatched_layers.append((k, model_dict[k].shape, v.shape))
        else:
            mismatched_layers.append((k, None, v.shape))

    model_dict.update(matched_dict)
    model.load_state_dict(model_dict, strict=False)

    logger = logging.getLogger(__name__)
    logger.info(f"Loaded layers: {list(matched_dict.keys())}")
    logger.info(f"Skipped layers (mismatch or missing):")
    for k, model_shape, pretrained_shape in mismatched_layers:
        logger.info(f" - {k}: model_shape={model_shape}, pretrained_shape={pretrained_shape}")
    
    return model
