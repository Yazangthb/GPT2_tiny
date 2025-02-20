# utils.py
import math
import os
import torch

def get_lr(it, max_lr=6e-4, min_lr=6e-4*0.1, warmup_steps=10, max_steps=50):
    """
    Cosine decay learning rate schedule with a warmup period.
    """
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    if it > max_steps:
        return min_lr
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

def save_checkpoint(model, optimizer, step, checkpoint_dir):
    """
    Saves a checkpoint containing the current training step, model, and optimizer states.
    """
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_step_{step}.pt")
    state = {
        "step": step,
        "model_state_dict": model.module.state_dict() if hasattr(model, "module") else model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict()
    }
    torch.save(state, checkpoint_path)
    print(f"Checkpoint saved at {checkpoint_path}")

def load_checkpoint(model, optimizer, checkpoint_path, device):
    """
    Loads a checkpoint and returns the step to resume training from.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_state_dict = checkpoint["model_state_dict"]
    model.load_state_dict(model_state_dict)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_step = checkpoint["step"]
    print(f"Resumed from checkpoint {checkpoint_path} at step {start_step}")
    return start_step
