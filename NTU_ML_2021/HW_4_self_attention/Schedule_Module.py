'''
Define various learning rate schedulers.
Currently, only the cosine scheduler with warmup is implemented.
'''

import math
import numpy as np
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def get_cosine_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
    ):
    # -------------------------
    # 🔒 Input validation
    # -------------------------
    if num_training_steps <= 0:
        raise ValueError(f"num_training_steps must be > 0, num_training_steps = {num_training_steps}")

    if num_warmup_steps < 0:
        raise ValueError(f"num_warmup_steps must be >= 0, num_warmup_steps = {num_warmup_steps}")

    if num_warmup_steps > num_training_steps:
        raise ValueError(
            f"num_warmup_steps ({num_warmup_steps}) cannot exceed "
            f"num_training_steps ({num_training_steps})"
        )

    if num_cycles < 1e-10:
        raise ValueError(f"num_cycles must be > 1e-20, num_cycles = {num_cycles}")

    if optimizer is None:
        raise ValueError("optimizer must not be None")

    def lr_lambda(current_step):
        # Warmup: linearly increase the learning rate from 0 to the initial lr set in the optimizer.
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        
        # Cosine decay: after the warmup phase, the learning rate will decrease following a cosine curve until it reaches 0 at the end of training.
        # lr = 0.5 * (lr_max - lr_min) * (1 + cos(2 * pi * num_cycles * progress)) + lr_min
        # progress = (current_step - num_warmup_steps) / (num_training_steps - num_warmup_steps)
        # progress = t/T. Here, t = current_step - num_warmup_steps and T is the total number of training steps after warmup (total_step - warmup_step).
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        # Clamp progress to ensure it is between 0 and 1.
        progress = min(1.0, max(0.0, progress))
        ncycles = max(1e-20, num_cycles)  # Ensure num_cycles is non-negative.
        return max(0.0, 0.5 * (1.0 + math.cos(2.0 * np.pi * ncycles * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)