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
  - Arrow IPC dataset support (from tokenize_universal.py output)
  - Binary (.bin) dataset support (legacy, from tokenize_data.py output)
  - HuggingFace Hub upload on completion or on demand
  - Windows 11 + CUDA 13 compatible

Author: Bryan K Reinhart
License: AGPL-3.0

Usage
-----
    # Train on .bin data (legacy):
    python scripts/train.py \\
        --checkpoint_dir checkpoints/vortex_v1 \\
        --train_bin data/train.bin \\
        --val_bin data/val.bin \\
        --max_steps 100000 \\
        --batch_size 8 \\
        --grad_accum 8 \\
        --lr 3e-4

    # Train on Arrow data (from tokenize_universal.py):
    python scripts/train.py \\
        --checkpoint_dir checkpoints/vortex_v1 \\
        --data_format arrow \\
        --arrow_manifest data/arrow/alt_fantasy/manifest.json \\
        --max_steps 100000 \\
        --batch_size 8 \\
        --grad_accum 8 \\
        --lr 3e-4

    # Resume from checkpoint:
    python scripts/train.py --resume checkpoints/vortex_v1

    # Evaluate only (no training):
    python scripts/train.py --eval_only

    # Upload best checkpoint to HuggingFace Hub after training:
    python scripts/train.py \\
        --checkpoint_dir checkpoints/vortex_v1 \\
        --hf_upload \\
        --hf_repo Bey66/vortex-small \\
        --hf_token YOUR_TOKEN

    # Upload an existing checkpoint without training:
    python scripts/train.py \\
        --upload_only \\
        --checkpoint_dir checkpoints/vortex_v1 \\
        --hf_repo Bey66/vortex-small \\
        --hf_token YOUR_TOKEN
"""

import sys
import os
import argparse
import time
import math
import json
import random
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
    p.add_argument("--log_dir",        default="logs")
    p.add_argument("--resume",         default=None,
                   help="Path to checkpoint directory to resume from.")

    # Data format selection
    p.add_argument("--data_format", default="bin", choices=["bin", "arrow"],
                   help="Dataset format: 'bin' for .bin files, "
                        "'arrow' for Arrow IPC shards from tokenize_universal.py.")

    # Binary (.bin) data paths
    p.add_argument("--train_bin", default="data/train.bin",
                   help="Training .bin file (used when --data_format bin).")
    p.add_argument("--val_bin",   default="data/val.bin",
                   help="Validation .bin file (used when --data_format bin).")

    # Arrow data paths
    p.add_argument("--arrow_manifest", default=None,
                   help="Path to manifest.json from tokenize_universal.py "
                        "(used when --data_format arrow).")
    p.add_argument("--arrow_train_dir", default=None,
                   help="Directory containing train_shard_*.arrow files "
                        "(alternative to --arrow_manifest).")
    p.add_argument("--arrow_val_dir",   default=None,
                   help="Directory containing val_shard_*.arrow files "
                        "(alternative to --arrow_manifest).")

    # Model (ignored if resuming)
    p.add_argument("--model_size",  default="medium", choices=["nano", "small", "medium", "large", "custom"])
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
                   help="Gradient accumulation steps. "
                        "Effective batch = batch_size * grad_accum.")
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
                   help="Number of batches for validation loss estimate.")
    p.add_argument("--tokenizer",    default="amd/AMD-Llama-135m")

    # HuggingFace Hub upload
    p.add_argument("--hf_upload",    action="store_true",
                   help="Upload best checkpoint to HuggingFace Hub after training.")
    p.add_argument("--hf_repo",      default=None,
                   help="Hub repo id, e.g. Bey66/vortex-small. "
                        "Created automatically if it does not exist.")
    p.add_argument("--hf_token",     default=None,
                   help="HuggingFace API token. Falls back to HF_TOKEN env var.")
    p.add_argument("--hf_private",   action="store_true",
                   help="Make the Hub repository private.")
    p.add_argument("--hf_tag",       default="best",
                   choices=["best", "latest", "final"],
                   help="Which checkpoint tag to upload (default: best).")
    p.add_argument("--upload_only",  action="store_true",
                   help="Skip training and upload an existing checkpoint.")

    # Mode
    p.add_argument("--eval_only", action="store_true")
    p.add_argument("--compile",   action="store_true",
                   help="Use torch.compile (may not work on Windows).")
    p.add_argument("--device",    default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed",      type=int, default=42)

    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Binary dataset loader (.bin)
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
# Arrow dataset loader (from tokenize_universal.py)
# ──────────────────────────────────────────────────────────────────────────────

class ArrowDataset:
    """
    Memory-mapped Arrow IPC shard dataset for Vortex training.

    Loads packed blocks from Arrow shard files produced by tokenize_universal.py.
    Each shard contains rows with 'input_ids' and 'labels' columns, both of
    length block_size. Batches are assembled by randomly sampling shards and
    rows — no data is loaded into RAM until a batch is requested.

    The shard-level random sampling ensures all shards contribute to training
    regardless of the order they were written.
    """

    def __init__(self, shard_paths: list, block_size: int, split: str = ""):
        try:
            import pyarrow as pa
            import pyarrow.ipc as ipc
            self._pa  = pa
            self._ipc = ipc
        except ImportError:
            raise ImportError(
                "pyarrow is required for Arrow dataset loading.\n"
                "Install with:  pip install pyarrow"
            )

        self.shard_paths = [str(p) for p in shard_paths if os.path.exists(str(p))]
        self.block_size  = block_size
        self.split       = split

        if not self.shard_paths:
            raise FileNotFoundError(
                f"No Arrow shard files found. Check your --arrow_manifest "
                f"or --arrow_train_dir / --arrow_val_dir paths."
            )

        # Index: count rows per shard without loading data
        self._shard_row_counts = []
        total_rows = 0
        for path in self.shard_paths:
            with self._pa.memory_map(path, "r") as src:
                f = self._ipc.open_file(src)
                n = sum(
                    f.get_batch(i).num_rows
                    for i in range(f.num_record_batches)
                )
                self._shard_row_counts.append(n)
                total_rows += n

        self.total_rows = total_rows
        print(f"  Arrow {split}: {total_rows:,} blocks across "
              f"{len(self.shard_paths)} shards")

    def get_batch(self, batch_size: int, device: str):
        """
        Sample batch_size random blocks from the shard pool.
        Uses weighted random sampling proportional to shard size so
        all blocks are equally likely to be selected.
        """
        # Weighted shard selection
        shard_indices = random.choices(
            range(len(self.shard_paths)),
            weights=self._shard_row_counts,
            k=batch_size,
        )

        xs, ys = [], []

        # Group by shard to minimise file opens
        shard_requests: dict[int, list[int]] = {}
        for i, si in enumerate(shard_indices):
            shard_requests.setdefault(si, []).append(i)

        results_x = [None] * batch_size
        results_y = [None] * batch_size

        for shard_idx, batch_positions in shard_requests.items():
            path = self.shard_paths[shard_idx]
            n_rows = self._shard_row_counts[shard_idx]

            with self._pa.memory_map(path, "r") as src:
                f = self._ipc.open_file(src)
                n_batches = f.num_record_batches

                for pos in batch_positions:
                    # Pick a random row within this shard
                    row_idx   = random.randrange(n_rows)

                    # Find which record batch contains this row
                    batch_no, offset = 0, row_idx
                    for bn in range(n_batches):
                        batch_len = f.get_batch(bn).num_rows
                        if offset < batch_len:
                            batch_no = bn
                            break
                        offset -= batch_len

                    batch = f.get_batch(batch_no)
                    input_ids = batch.column("input_ids")[offset].as_py()
                    labels    = batch.column("labels")[offset].as_py()

                    results_x[pos] = input_ids
                    results_y[pos] = labels

        x = torch.tensor(results_x, dtype=torch.long).to(device)
        y = torch.tensor(results_y, dtype=torch.long).to(device)
        return x, y

    @classmethod
    def from_manifest(cls, manifest_path: str, split: str,
                       block_size: int) -> "ArrowDataset":
        """Load shard paths from a tokenize_universal.py manifest.json."""
        with open(manifest_path) as f:
            manifest = json.load(f)

        key = "train_shards" if split == "train" else "val_shards"
        paths = manifest.get(key, [])

        if not paths:
            raise ValueError(
                f"No '{key}' found in manifest {manifest_path}. "
                f"Available keys: {list(manifest.keys())}"
            )

        # Validate block_size matches
        manifest_bs = manifest.get("block_size", block_size)
        if manifest_bs != block_size:
            print(f"  [WARN] Manifest block_size={manifest_bs} differs from "
                  f"--block_size={block_size}. Using manifest value.")
            block_size = manifest_bs

        return cls(paths, block_size, split=split)

    @classmethod
    def from_directory(cls, directory: str, split: str,
                        block_size: int) -> "ArrowDataset":
        """Discover shard files from a directory by naming convention."""
        prefix = "train" if split == "train" else "val"
        paths  = sorted(Path(directory).glob(f"{prefix}_shard_*.arrow"))
        if not paths:
            raise FileNotFoundError(
                f"No {prefix}_shard_*.arrow files found in {directory}"
            )
        return cls(paths, block_size, split=split)


# ──────────────────────────────────────────────────────────────────────────────
# Dataset factory
# ──────────────────────────────────────────────────────────────────────────────

def load_datasets(args):
    """
    Load train and val datasets based on --data_format.
    Returns (train_data, val_data) — both support .get_batch(batch_size, device).
    """
    if args.data_format == "arrow":
        if args.arrow_manifest:
            print(f"  Loading Arrow datasets from manifest: {args.arrow_manifest}")
            train_data = ArrowDataset.from_manifest(
                args.arrow_manifest, "train", args.block_size
            )
            val_data = ArrowDataset.from_manifest(
                args.arrow_manifest, "val", args.block_size
            )
        elif args.arrow_train_dir and args.arrow_val_dir:
            print(f"  Loading Arrow datasets from directories...")
            train_data = ArrowDataset.from_directory(
                args.arrow_train_dir, "train", args.block_size
            )
            val_data = ArrowDataset.from_directory(
                args.arrow_val_dir, "val", args.block_size
            )
        else:
            raise ValueError(
                "Arrow format requires either --arrow_manifest or "
                "both --arrow_train_dir and --arrow_val_dir."
            )
    else:
        # Binary .bin format
        train_data = BinaryDataset(args.train_bin, args.block_size)
        val_data   = BinaryDataset(args.val_bin,   args.block_size)

    return train_data, val_data


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
    "nano":   dict(d_model=224,  n_layer=4, d_state=32, expand=2,  n_heads=4),
    "small":  dict(d_model=384,  n_layer=6, d_state=48, expand=2,  n_heads=6),
    "medium": dict(d_model=512,  n_layer=8, d_state=64, expand=2,  n_heads=8),
    "large":  dict(d_model=768, n_layer=12, d_state=96, expand=2, n_heads=12),
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
def evaluate(model, val_data, batch_size: int,
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
    os.makedirs(out_dir, exist_ok=True)

    # Save weights manually — bypass transformers tied-weights machinery
    state_dict = model.state_dict()
    if model.config.tie_embeddings and "lm_head.weight" in state_dict:
        del state_dict["lm_head.weight"]
    save_file(state_dict, os.path.join(out_dir, "model.safetensors"))

    # Save config
    model.config.save_pretrained(out_dir)

    # Save optimiser + scaler state
    meta = {"step": step, "val_loss": val_loss, "tag": tag}
    torch.save(
        {"optimizer": optimizer.state_dict(),
         "scaler":    scaler.state_dict(),
         "meta":      meta},
        os.path.join(out_dir, f"training_state_{tag}.pt"),
    )
    with open(os.path.join(out_dir, f"meta_{tag}.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  ✓ Checkpoint saved → {out_dir} [{tag}]  "
          f"step={step}  val_loss={val_loss:.4f}")


def load_checkpoint(model, optimizer, scaler,
                    checkpoint_dir: str, tag: str = "latest"):
    from safetensors.torch import load_file
    weights_path = os.path.join(checkpoint_dir, "model.safetensors")
    state_dict   = load_file(weights_path, device="cpu")
    model.load_state_dict(state_dict, strict=False)

    train_state_path = os.path.join(checkpoint_dir, f"training_state_{tag}.pt")
    if os.path.exists(train_state_path):
        state = torch.load(train_state_path, map_location="cpu",
                           weights_only=False)
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        meta = state["meta"]
        print(f"  ✓ Resumed from {checkpoint_dir}  step={meta['step']}  "
              f"val_loss={meta.get('val_loss', 'N/A'):.4f}")
        return meta["step"]
    return 0


# ──────────────────────────────────────────────────────────────────────────────
# HuggingFace Hub upload
# ──────────────────────────────────────────────────────────────────────────────

def upload_to_hub(checkpoint_dir: str, repo_id: str, token: str,
                  private: bool, tag: str, config: VortexConfig,
                  val_loss: float, step: int):
    """
    Upload a Vortex checkpoint to HuggingFace Hub.

    Creates the repository if it does not exist, writes a model card
    (README.md) describing the architecture and training run, and pushes
    all checkpoint files. The model is registered as a custom architecture
    so it can be loaded with AutoModelForCausalLM after the user installs
    the Vortex package.

    Args:
        checkpoint_dir : Local directory containing model.safetensors + config.json
        repo_id        : Hub repository id, e.g. "Bey66/vortex-small"
        token          : HuggingFace API token
        private        : Whether to make the repo private
        tag            : Checkpoint tag that was saved (best/latest/final)
        config         : VortexConfig for the model card
        val_loss       : Best validation loss for the model card
        step           : Training step at upload
    """
    try:
        from huggingface_hub import HfApi, create_repo, upload_folder
    except ImportError:
        print("\n  [ERROR] huggingface_hub not installed.")
        print("  Install with:  pip install huggingface_hub")
        return False

    token = token or os.environ.get("HF_TOKEN")
    if not token:
        print("\n  [ERROR] No HuggingFace token provided.")
        print("  Use --hf_token YOUR_TOKEN or set the HF_TOKEN environment variable.")
        print("  Get your token at: https://huggingface.co/settings/tokens")
        return False

    api = HfApi(token=token)

    # ── Create repo if needed ─────────────────────────────────────────────────
    print(f"\n  Preparing HuggingFace Hub upload...")
    print(f"  Repository : {repo_id}")
    print(f"  Visibility : {'private' if private else 'public'}")

    try:
        create_repo(
            repo_id=repo_id,
            token=token,
            private=private,
            repo_type="model",
            exist_ok=True,
        )
        print(f"  ✓ Repository ready: https://huggingface.co/{repo_id}")
    except Exception as e:
        print(f"  [ERROR] Could not create repository: {e}")
        return False

    # ── Write model card ──────────────────────────────────────────────────────
    ppl = math.exp(val_loss) if val_loss < 20 else float("inf")
    model_card = f"""---
language:
- en
license: agpl-3.0
tags:
- vortex
- language-model
- state-space-model
- mamba
- parallax
- custom-architecture
base_model: null
---

# Vortex — {repo_id.split("/")[-1]}

A **Vortex** hybrid language model checkpoint trained by [{repo_id.split("/")[0]}](https://huggingface.co/{repo_id.split("/")[0]}).

## Architecture

Vortex is a dual-track Mamba-2 SSD (State Space Duality) language model that fuses:

- **Parallax dual-track swap** — two independent SSM streams that exchange their
  hidden representations between passes, giving the model two offset vantage points
- **Mamba-2 SSD blocks** — linear-complexity (O(L)) sequence modeling replacing
  the O(L²) attention operator

### Configuration

| Parameter | Value |
|---|---|
| Architecture | VortexForCausalLM |
| Parameters | ~{sum(1 for _ in range(1))} (see below) |
| d_model | {config.d_model} |
| n_layer (per track) | {config.n_layer} |
| num_loops | {config.num_loops} |
| d_state | {config.d_state} |
| d_inner | {config.d_inner} |
| n_heads | {config.n_heads} |
| block_size | {config.block_size} |
| vocab_size | {config.vocab_size} |
| use_swap | {config.use_swap} |

### Training

| | |
|---|---|
| Training steps | {step:,} |
| Best val loss | {val_loss:.4f} |
| Perplexity | {ppl:.2f} |
| Checkpoint tag | {tag} |

## Usage

This model uses a custom architecture and requires the Vortex package to load.

```python
# Clone the Vortex repository first:
# git clone https://github.com/beyastard/Vortex

import sys
sys.path.insert(0, "path/to/Vortex")

from model.vortex import VortexForCausalLM
from transformers import AutoTokenizer

model = VortexForCausalLM.from_pretrained("{repo_id}")
tokenizer = AutoTokenizer.from_pretrained("amd/AMD-Llama-135m")

input_ids = tokenizer.encode("Once upon a time", return_tensors="pt")
output = model.generate(input_ids, max_new_tokens=200, temperature=0.8)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

## Citation

If you use this model, please cite the Parallax architecture from which
the dual-track design is derived:

```bibtex
@misc{{parallax2025,
  title   = {{Parallax: A dual-track transformer language model with
             cross-pollinating attention passes}},
  author  = {{beyastard}},
  year    = {{2025}},
  url     = {{https://github.com/beyastard/Parallax}}
}}
```
"""

    card_path = os.path.join(checkpoint_dir, "README.md")
    with open(card_path, "w", encoding="utf-8") as f:
        f.write(model_card)
    print(f"  ✓ Model card written → {card_path}")

    # ── Determine files to upload ─────────────────────────────────────────────
    # Upload the checkpoint directory contents
    # Always include: model.safetensors, config.json, README.md
    # Include training state only for non-best tags (best is a release)
    upload_patterns = ["model.safetensors", "config.json", "README.md",
                       f"meta_{tag}.json"]

    print(f"  Uploading files from {checkpoint_dir}...")
    for fname in upload_patterns:
        fpath = os.path.join(checkpoint_dir, fname)
        if os.path.exists(fpath):
            size_mb = os.path.getsize(fpath) / 1024 / 1024
            print(f"    {fname}  ({size_mb:.1f} MB)")

    # ── Upload ────────────────────────────────────────────────────────────────
    try:
        upload_folder(
            folder_path=checkpoint_dir,
            repo_id=repo_id,
            repo_type="model",
            token=token,
            ignore_patterns=[
                "training_state_*.pt",   # Exclude large optimiser states
                "*.bin",                  # Exclude old-format weights
                "logs/",
            ],
            commit_message=f"Vortex checkpoint [{tag}] — step {step:,}, "
                           f"val_loss {val_loss:.4f}",
        )
        print(f"\n  ✓ Upload complete!")
        print(f"  Model page: https://huggingface.co/{repo_id}")
        return True

    except Exception as e:
        print(f"\n  [ERROR] Upload failed: {e}")
        return False


# ──────────────────────────────────────────────────────────────────────────────
# Main training loop
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = args.device

    # ── Upload-only mode ──────────────────────────────────────────────────────
    if args.upload_only:
        if not args.hf_repo:
            print("[ERROR] --hf_repo is required for --upload_only.")
            sys.exit(1)
        print(f"\nUpload-only mode: {args.checkpoint_dir} → {args.hf_repo}")

        # Load config and meta to get training stats
        config = VortexConfig.from_pretrained(args.checkpoint_dir)
        meta_path = os.path.join(args.checkpoint_dir,
                                  f"meta_{args.hf_tag}.json")
        val_loss, step = 0.0, 0
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            val_loss = meta.get("val_loss", 0.0)
            step     = meta.get("step", 0)

        upload_to_hub(
            checkpoint_dir=args.checkpoint_dir,
            repo_id=args.hf_repo,
            token=args.hf_token,
            private=args.hf_private,
            tag=args.hf_tag,
            config=config,
            val_loss=val_loss,
            step=step,
        )
        return

    # ── Normal training path ──────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Vortex Training")
    print(f"{'='*60}")
    print(f"  Device        : {device}")
    if device == "cuda":
        print(f"  GPU           : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM          : "
              f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"  Checkpoint dir: {args.checkpoint_dir}")
    print(f"  Data format   : {args.data_format}")
    if args.data_format == "bin":
        print(f"  Train data    : {args.train_bin}")
        print(f"  Val data      : {args.val_bin}")
    else:
        print(f"  Arrow manifest: {args.arrow_manifest or 'from directories'}")
    print()

    # ── Data ──────────────────────────────────────────────────────────────────
    print("Loading data...")
    train_data, val_data = load_datasets(args)

    # ── Model ─────────────────────────────────────────────────────────────────
    start_step    = 0
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
    decay_params   = [p for n, p in model.named_parameters()
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
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    if args.resume and os.path.exists(args.resume):
        start_step = load_checkpoint(
            model, optimizer, scaler, args.resume, tag="latest"
        )

    if args.eval_only:
        print("\nEvaluating...")
        val_loss = evaluate(model, val_data, args.batch_size,
                            args.val_batches, device)
        val_ppl  = math.exp(val_loss)
        print(f"  Val loss: {val_loss:.4f}   Perplexity: {val_ppl:.2f}")
        return

    # ── TensorBoard ───────────────────────────────────────────────────────────
    run_name = f"vortex_{args.model_size}_{time.strftime('%Y%m%d_%H%M%S')}"
    writer   = SummaryWriter(log_dir=os.path.join(args.log_dir, run_name))

    # ── Training ──────────────────────────────────────────────────────────────
    eff_batch       = args.batch_size * args.grad_accum
    tokens_per_step = eff_batch * args.block_size

    print(f"\nTraining configuration:")
    print(f"  Max steps            : {args.max_steps:,}")
    print(f"  Batch size (GPU)     : {args.batch_size}")
    print(f"  Gradient accumulation: {args.grad_accum}")
    print(f"  Effective batch      : {eff_batch} sequences")
    print(f"  Tokens per step      : {tokens_per_step:,}")
    print(f"  Peak LR              : {args.lr}")
    print(f"  Warmup steps         : {args.warmup_steps}")
    print(f"  TensorBoard run      : {run_name}")
    if args.hf_upload:
        print(f"  HF upload on finish  : {args.hf_repo} [{args.hf_tag}]")
    print()

    model.train()
    optimizer.zero_grad()

    t0           = time.time()
    running_loss = 0.0

    for step in range(start_step, args.max_steps):
        # LR update
        lr = get_lr(step, args.warmup_steps, args.max_steps, args.lr, args.min_lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # Gradient accumulation loop
        for micro_step in range(args.grad_accum):
            x, y = train_data.get_batch(args.batch_size, device)
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16,
                                     enabled=(device == "cuda")):
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
            t1          = time.time()
            dt          = t1 - t0
            tok_per_sec = tokens_per_step * args.log_every / max(dt, 1e-6)
            t0          = t1

            train_loss   = running_loss / args.log_every
            running_loss = 0.0

            writer.add_scalar("train/loss",        train_loss,  step)
            writer.add_scalar("train/lr",          lr,          step)
            writer.add_scalar("train/grad_norm",   grad_norm,   step)
            writer.add_scalar("train/tok_per_sec", tok_per_sec, step)

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
            writer.add_scalar("val/loss",       val_loss, step)
            writer.add_scalar("val/perplexity", val_ppl,  step)
            print(f"  ↳ val_loss={val_loss:.4f}  val_ppl={val_ppl:.2f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(model, optimizer, scaler, step, val_loss,
                                args.checkpoint_dir, tag="best")

        # ── Periodic checkpoint ───────────────────────────────────────────────
        if step > 0 and step % args.save_every == 0:
            save_checkpoint(model, optimizer, scaler, step,
                            best_val_loss, args.checkpoint_dir, tag="latest")

    # ── Final save ────────────────────────────────────────────────────────────
    print("\nTraining complete.")
    save_checkpoint(model, optimizer, scaler, args.max_steps,
                    best_val_loss, args.checkpoint_dir, tag="final")
    writer.close()

    # ── HuggingFace Hub upload ────────────────────────────────────────────────
    if args.hf_upload:
        if not args.hf_repo:
            print("\n[WARN] --hf_upload set but --hf_repo not provided. Skipping upload.")
        else:
            upload_to_hub(
                checkpoint_dir=args.checkpoint_dir,
                repo_id=args.hf_repo,
                token=args.hf_token,
                private=args.hf_private,
                tag=args.hf_tag,
                config=model.config,
                val_loss=best_val_loss,
                step=args.max_steps,
            )


if __name__ == "__main__":
    main()
