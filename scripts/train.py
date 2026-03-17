"""
train.py
========
Training script for the Vortex hybrid language model.

Features
--------
  - Mixed-precision training (FP16 AMP via torch.amp)
  - Gradient accumulation for effective large-batch training on limited VRAM
  - Cosine LR schedule with linear warm-up
  - Gradient clipping
  - Periodic validation loss evaluation
  - TensorBoard logging (loss, lr, grad_norm, tokens/sec)
  - Checkpoint saving: best model (by val loss) + latest step
  - Resume from checkpoint
  - Windows 11 + CUDA 13 compatible

Author: Bryan K Reinhart
License: AGPL-3.0

Usage
-----
    python scripts/train.py

    # With custom config:
    python scripts/train.py \\
        --checkpoint_dir checkpoints/vortex_v1 \\
        --train_bin data/train.bin \\
        --val_bin data/val.bin \\
        --max_steps 100000 \\
        --batch_size 8 \\
        --grad_accum 8 \\
        --lr 3e-4

    # Resume from checkpoint:
    python scripts/train.py --resume checkpoints/vortex_v1

    # Evaluate only (no training):
    python scripts/train.py --eval_only
"""

import sys
import os
import argparse
import time
import math
import json
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from safetensors.torch import save_file
from transformers import AutoTokenizer

from model.vortex import VortexConfig, VortexForCausalLM


# ──────────────────────────────────────────────────────────────────────────────
# Args
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train Vortex language model.")

    # Paths
    p.add_argument("--checkpoint_dir", default="checkpoints/vortex_v1")
    p.add_argument("--train_bin",      default="data/train.bin")
    p.add_argument("--val_bin",        default="data/val.bin")
    p.add_argument("--log_dir",        default="logs")
    p.add_argument("--resume",         default=None,
                   help="Path to checkpoint directory to resume from.")

    # Model (ignored if resuming)
    p.add_argument("--model_size",  default="medium", choices=["small", "medium", "custom"])
    p.add_argument("--d_model",     type=int, default=512)
    p.add_argument("--n_layer",     type=int, default=8)
    p.add_argument("--d_state",     type=int, default=64)
    p.add_argument("--expand",      type=int, default=2)
    p.add_argument("--n_heads",     type=int, default=8)
    p.add_argument("--num_loops",   type=int, default=2)
    p.add_argument("--block_size",  type=int, default=1024)
    p.add_argument("--vocab_size",  type=int, default=32000)
    p.add_argument("--no_triton",   action="store_true")

    # Training
    p.add_argument("--max_steps",      type=int,   default=100_000)
    p.add_argument("--batch_size",     type=int,   default=8,
                   help="Sequences per GPU step (before gradient accumulation).")
    p.add_argument("--grad_accum",     type=int,   default=8,
                   help="Gradient accumulation steps. Effective batch = batch_size * grad_accum.")
    p.add_argument("--lr",             type=float, default=3e-4)
    p.add_argument("--min_lr",         type=float, default=3e-5,
                   help="Minimum LR at end of cosine decay.")
    p.add_argument("--warmup_steps",   type=int,   default=2000)
    p.add_argument("--weight_decay",   type=float, default=0.1)
    p.add_argument("--grad_clip",      type=float, default=1.0)
    p.add_argument("--dropout",        type=float, default=0.0)
    p.add_argument("--beta1",          type=float, default=0.9)
    p.add_argument("--beta2",          type=float, default=0.95)
    p.add_argument("--eps",            type=float, default=1e-8)

    # Logging / saving
    p.add_argument("--log_every",    type=int, default=100)
    p.add_argument("--val_every",    type=int, default=1000)
    p.add_argument("--save_every",   type=int, default=5000)
    p.add_argument("--val_batches",  type=int, default=50,
                   help="Number of batches to use for validation loss estimate.")
    p.add_argument("--tokenizer",    default="gpt2")

    # Mode
    p.add_argument("--eval_only", action="store_true")
    p.add_argument("--compile",   action="store_true",
                   help="Use torch.compile (requires PyTorch 2.x, may not work on Windows).")
    p.add_argument("--device",    default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed",      type=int, default=42)

    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Data loader
# ──────────────────────────────────────────────────────────────────────────────

class BinaryDataset:
    """
    Memory-mapped loader over a packed uint32 binary token file.
    Returns random (block_size+1)-length windows for causal LM training.
    """

    def __init__(self, path: str, block_size: int):
        self.block_size = block_size
        self.data = np.memmap(path, dtype=np.uint32, mode="r")
        n = len(self.data)
        assert n > block_size, f"Dataset too small: {n} tokens, need > {block_size}"
        self.n = n
        print(f"  Loaded {path}: {n:,} tokens ({n * 4 / 1e9:.3f} GB)")

    def get_batch(self, batch_size: int, device: str):
        ix = np.random.randint(0, self.n - self.block_size - 1, size=batch_size)
        x = np.stack([self.data[i : i + self.block_size].astype(np.int64) for i in ix])
        y = np.stack([self.data[i + 1 : i + self.block_size + 1].astype(np.int64) for i in ix])
        x = torch.from_numpy(x).to(device)
        y = torch.from_numpy(y).to(device)
        return x, y


# ──────────────────────────────────────────────────────────────────────────────
# LR schedule
# ──────────────────────────────────────────────────────────────────────────────

def get_lr(step: int, warmup_steps: int, max_steps: int,
           lr: float, min_lr: float) -> float:
    """Cosine decay with linear warm-up."""
    if step < warmup_steps:
        return lr * step / max(warmup_steps, 1)
    if step >= max_steps:
        return min_lr
    progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
    decay = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + decay * (lr - min_lr)


# ──────────────────────────────────────────────────────────────────────────────
# Build model
# ──────────────────────────────────────────────────────────────────────────────

SIZE_PRESETS = {
    "small":  dict(d_model=384, n_layer=6, d_state=48, expand=2, n_heads=6),
    "medium": dict(d_model=512, n_layer=8, d_state=64, expand=2, n_heads=8),
}

def build_model(args) -> VortexForCausalLM:
    if args.model_size in SIZE_PRESETS:
        kw = SIZE_PRESETS[args.model_size]
    else:
        kw = dict(d_model=args.d_model, n_layer=args.n_layer,
                  d_state=args.d_state, expand=args.expand, n_heads=args.n_heads)
    config = VortexConfig(
        vocab_size=args.vocab_size,
        num_loops=args.num_loops,
        block_size=args.block_size,
        dropout=args.dropout,
        use_triton=not args.no_triton,
        **kw,
    )
    return VortexForCausalLM(config)


# ──────────────────────────────────────────────────────────────────────────────
# Validation
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, val_data: BinaryDataset, batch_size: int,
             val_batches: int, device: str) -> float:
    model.eval()
    losses = []
    for _ in range(val_batches):
        x, y = val_data.get_batch(batch_size, device)
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16,
                                 enabled=(device == "cuda")):
            out = model(input_ids=x, labels=y)
        losses.append(out.loss.item())
    model.train()
    return float(np.mean(losses))


# ──────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ──────────────────────────────────────────────────────────────────────────────

def save_checkpoint(model, optimizer, scaler, step: int, val_loss: float,
                    out_dir: str, tag: str = "latest"):
    import json
    from safetensors.torch import save_file

    os.makedirs(out_dir, exist_ok=True)

    # Save weights manually, de-duplicating tied embedding/lm_head
    state_dict = model.state_dict()
    if model.config.tie_embeddings and "lm_head.weight" in state_dict:
        del state_dict["lm_head.weight"]
    save_file(state_dict, os.path.join(out_dir, "model.safetensors"))

    # Save config
    model.config.save_pretrained(out_dir)

    # Save optimizer + scaler state
    meta = {"step": step, "val_loss": val_loss, "tag": tag}
    torch.save(
        {"optimizer": optimizer.state_dict(),
         "scaler":    scaler.state_dict(),
         "meta":      meta},
        os.path.join(out_dir, f"training_state_{tag}.pt"),
    )
    with open(os.path.join(out_dir, f"meta_{tag}.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  ✓ Checkpoint saved → {out_dir} [{tag}]  step={step}  val_loss={val_loss:.4f}")


def load_checkpoint(model, optimizer, scaler, checkpoint_dir: str, tag: str = "latest"):
    from safetensors.torch import load_file
    weights_path = os.path.join(checkpoint_dir, "model.safetensors")
    state_dict = load_file(weights_path, device="cpu")
    model.load_state_dict(state_dict, strict=False)

    train_state_path = os.path.join(checkpoint_dir, f"training_state_{tag}.pt")
    if os.path.exists(train_state_path):
        state = torch.load(train_state_path, map_location="cpu")
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        meta = state["meta"]
        print(f"  ✓ Resumed from {checkpoint_dir}  step={meta['step']}  "
              f"val_loss={meta.get('val_loss', 'N/A'):.4f}")
        return meta["step"]
    return 0


# ──────────────────────────────────────────────────────────────────────────────
# Main training loop
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = args.device
    print(f"\n{'='*60}")
    print(f"Vortex Training")
    print(f"{'='*60}")
    print(f"  Device        : {device}")
    if device == "cuda":
        print(f"  GPU           : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM          : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"  Checkpoint dir: {args.checkpoint_dir}")
    print(f"  Train data    : {args.train_bin}")
    print(f"  Val data      : {args.val_bin}")
    print()

    # ── Data ──────────────────────────────────────────────────────────────────
    print("Loading data...")
    train_data = BinaryDataset(args.train_bin, args.block_size)
    val_data   = BinaryDataset(args.val_bin,   args.block_size)

    # ── Model ─────────────────────────────────────────────────────────────────
    start_step = 0
    best_val_loss = float("inf")

    if args.resume and os.path.exists(args.resume):
        print(f"\nLoading model from {args.resume}...")
        model = VortexForCausalLM.from_pretrained(args.resume)
    else:
        print(f"\nBuilding new model (size={args.model_size})...")
        model = build_model(args)

    model = model.to(device)

    params = sum(p.numel() for p in set(model.parameters()))
    print(f"  Unique parameters: {params:,}  (~{params/1e6:.2f}M)")
    print(f"  Config: {model.config}")

    if args.compile:
        try:
            print("  Compiling model with torch.compile...")
            model = torch.compile(model)
        except Exception as e:
            print(f"  torch.compile failed ({e}), continuing without.")

    # ── Optimiser ─────────────────────────────────────────────────────────────
    # Separate weight-decay and no-decay groups
    decay_params  = [p for n, p in model.named_parameters()
                     if p.ndim >= 2 and p.requires_grad]
    nodecay_params = [p for n, p in model.named_parameters()
                      if p.ndim < 2 and p.requires_grad]
    param_groups = [
        {"params": decay_params,   "weight_decay": args.weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(
        param_groups, lr=args.lr,
        betas=(args.beta1, args.beta2), eps=args.eps,
        fused=(device == "cuda"),
    )
    scaler = torch.amp.GradScaler(enabled=(device == "cuda"))

    if args.resume and os.path.exists(args.resume):
        start_step = load_checkpoint(model, optimizer, scaler, args.resume, tag="latest")

    if args.eval_only:
        print("\nEvaluating...")
        val_loss = evaluate(model, val_data, args.batch_size, args.val_batches, device)
        val_ppl  = math.exp(val_loss)
        print(f"  Val loss: {val_loss:.4f}   Perplexity: {val_ppl:.2f}")
        return

    # ── TensorBoard ───────────────────────────────────────────────────────────
    run_name = f"vortex_{args.model_size}_{time.strftime('%Y%m%d_%H%M%S')}"
    writer   = SummaryWriter(log_dir=os.path.join(args.log_dir, run_name))

    # ── Training ──────────────────────────────────────────────────────────────
    eff_batch = args.batch_size * args.grad_accum
    tokens_per_step = eff_batch * args.block_size

    print(f"\nTraining configuration:")
    print(f"  Max steps           : {args.max_steps:,}")
    print(f"  Batch size (GPU)    : {args.batch_size}")
    print(f"  Gradient accumulation: {args.grad_accum}")
    print(f"  Effective batch     : {eff_batch} sequences")
    print(f"  Tokens per step     : {tokens_per_step:,}")
    print(f"  Peak LR             : {args.lr}")
    print(f"  Warmup steps        : {args.warmup_steps}")
    print(f"  TensorBoard run     : {run_name}")
    print()

    model.train()
    optimizer.zero_grad()

    t0 = time.time()
    running_loss = 0.0

    for step in range(start_step, args.max_steps):
        # LR update
        lr = get_lr(step, args.warmup_steps, args.max_steps, args.lr, args.min_lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # Gradient accumulation loop
        for micro_step in range(args.grad_accum):
            x, y = train_data.get_batch(args.batch_size, device)
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=(device == "cuda")):
                out  = model(input_ids=x, labels=y)
                loss = out.loss / args.grad_accum

            scaler.scale(loss).backward()
            running_loss += loss.item()

        # Gradient clip + optimiser step
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), args.grad_clip
        )
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        # ── Logging ───────────────────────────────────────────────────────────
        if step % args.log_every == 0:
            t1 = time.time()
            dt = t1 - t0
            tok_per_sec = tokens_per_step * args.log_every / max(dt, 1e-6)
            t0 = t1

            #train_loss = running_loss * args.grad_accum / args.log_every
            train_loss = running_loss / args.log_every
            running_loss = 0.0

            writer.add_scalar("train/loss",       train_loss,  step)
            writer.add_scalar("train/lr",         lr,          step)
            writer.add_scalar("train/grad_norm",  grad_norm,   step)
            writer.add_scalar("train/tok_per_sec",tok_per_sec, step)

            print(
                f"step {step:7d}/{args.max_steps}  "
                f"loss={train_loss:.4f}  lr={lr:.2e}  "
                f"gnorm={grad_norm:.3f}  tok/s={tok_per_sec:,.0f}"
            )

        # ── Validation ────────────────────────────────────────────────────────
        if step > 0 and step % args.val_every == 0:
            val_loss = evaluate(model, val_data, args.batch_size,
                                args.val_batches, device)
            val_ppl = math.exp(val_loss)
            writer.add_scalar("val/loss",        val_loss, step)
            writer.add_scalar("val/perplexity",  val_ppl,  step)
            print(f"  ↳ val_loss={val_loss:.4f}  val_ppl={val_ppl:.2f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(model, optimizer, scaler, step, val_loss,
                                args.checkpoint_dir, tag="best")

        # ── Periodic checkpoint ───────────────────────────────────────────────
        if step > 0 and step % args.save_every == 0:
            save_checkpoint(model, optimizer, scaler, step,
                            best_val_loss, args.checkpoint_dir, tag="latest")

    # Final save
    print("\nTraining complete.")
    save_checkpoint(model, optimizer, scaler, args.max_steps,
                    best_val_loss, args.checkpoint_dir, tag="final")
    writer.close()


if __name__ == "__main__":
    main()
