# train.py
import os
import time
import torch
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F
import argparse

from model.gpt import GPT, GPTConfig
from data.dataloader import DataLoaderLite
from utils import get_lr, save_checkpoint, load_checkpoint
from config import (MODEL_CONFIG, TOTAL_BATCH_SIZE, BATCH_SIZE, SEQ_LENGTH,
                    MAX_LR, MIN_LR, WARMUP_STEPS, MAX_STEPS,
                    CHECKPOINT_DIR, CHECKPOINT_INTERVAL, EVAL_INTERVAL)

def evaluate(model, device, prompt="The meaning of life is", max_length=50):
    """
    Generates text given a prompt to evaluate the model.
    """
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    tokens = enc.encode(prompt)
    tokens = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        for _ in range(max_length):
            logits, _ = model(tokens)
            logits = logits[:, -1, :]
            probs = torch.softmax(logits, dim=-1)
            _, next_token = torch.topk(probs, 1)
            tokens = torch.cat((tokens, next_token), dim=1)
    model.train()
    decoded = enc.decode(tokens[0].tolist())
    return decoded

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume training")
    parser.add_argument("--pretrained", type=str, default=None,
                        help="Pretrained model type to initialize weights from (e.g., gpt2, gpt2-medium, gpt2-large, gpt2-xl)")
    args = parser.parse_args()

    # Distributed training setup
    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        assert torch.cuda.is_available(), "CUDA is required for DDP"
        init_process_group(backend="nccl")
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        device = f"cuda:{ddp_local_rank}"
        torch.cuda.set_device(device)
        master_process = (ddp_rank == 0)
    else:
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        master_process = True
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")

    device_type = "cuda" if device.startswith("cuda") else "cpu"

    torch.manual_seed(1337)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1337)

    grad_accum_steps = TOTAL_BATCH_SIZE // (BATCH_SIZE * SEQ_LENGTH * ddp_world_size)
    if master_process:
        print(f"Total desired batch size: {TOTAL_BATCH_SIZE}")
        print(f"Calculated gradient accumulation steps: {grad_accum_steps}")

    train_loader = DataLoaderLite(B=BATCH_SIZE, T=SEQ_LENGTH,
                                  process_rank=ddp_rank, num_processes=ddp_world_size)

    # Create model: resume checkpoint takes precedence, then pretrained option, otherwise random initialization.
    if args.resume is None:
        if args.pretrained is not None:
            # Load pretrained weights from Hugging Face
            model = GPT.from_pretrained(args.pretrained)
            if master_process:
                print(f"Initialized model with pretrained weights from '{args.pretrained}'")
        else:
            # Train from scratch
            config = GPTConfig(**MODEL_CONFIG)
            model = GPT(config)
            if master_process:
                print("Initialized model from scratch.")
    else:
        # For resuming, we'll create a model with the same config as a scratch run.
        config = GPTConfig(**MODEL_CONFIG)
        model = GPT(config)
    
    model.to(device)
    model.train()
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
    raw_model = model.module if ddp else model

    optimizer = raw_model.configure_optimizers(weight_decay=0.1,
                                                learning_rate=MAX_LR,
                                                device_type=device_type)

    start_step = 0
    if args.resume is not None:
        start_step = load_checkpoint(model, optimizer, args.resume, device)
    # Inside main(), before the training loop:
    scaler = torch.cuda.amp.GradScaler()

    for step in range(start_step, MAX_STEPS):
        start_time = time.time()
        optimizer.zero_grad()
        loss_accum = 0.0

        for micro_step in range(grad_accum_steps):
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=device_type, dtype=torch.float16):
                logits, loss = model(x, y)
            loss = loss / grad_accum_steps
            loss_accum += loss.detach()
            if ddp:
                model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
            # Scale the loss and call backward
            scaler.scale(loss).backward()

        if ddp:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
        
        # Unscale gradients and perform gradient clipping
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        # Step the optimizer using the scaler and update the scale factor
        scaler.step(optimizer)
        scaler.update()
        
        if device_type == "cuda":
            torch.cuda.synchronize()
        end_time = time.time()
        dt = (end_time - start_time) * 1000
        tokens_per_sec = (BATCH_SIZE * SEQ_LENGTH * grad_accum_steps * ddp_world_size) / (end_time - start_time)
        if master_process:
            print(f"Step {step:4d}: loss = {loss.item():.4f}, lr = {get_lr(step, max_lr=MAX_LR, min_lr=MIN_LR, warmup_steps=WARMUP_STEPS, max_steps=MAX_STEPS):.4e}, norm = {norm:.4f}, "
                f"time = {dt:.2f} ms, tokens/sec: {tokens_per_sec:.2f}")

            if step % EVAL_INTERVAL == 0:
                eval_text = evaluate(model, device)
                print(f"Evaluation at step {step}:\n{eval_text}\n")

            if step % CHECKPOINT_INTERVAL == 0:
                save_checkpoint(model, optimizer, step, CHECKPOINT_DIR)


    if ddp:
        destroy_process_group()

if __name__ == "__main__":
    main()
