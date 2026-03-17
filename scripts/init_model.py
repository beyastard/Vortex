"""
init_model.py
=============
Initialise a blank (randomly-weighted) Vortex model and save it to disk
in HuggingFace SafeTensors format.

Author: Bryan K Reinhart
License: AGPL-3.0

Usage
-----
    python scripts/init_model.py [--output checkpoints/vortex_v1] [--size small|medium|custom]

Options
-------
    --output   Directory to save the model and config (default: checkpoints/vortex_v1)
    --size     Preset size: small (~32M), medium (~48M), or custom (uses explicit flags)

Custom size flags (only used with --size custom):
    --d_model, --n_layer, --d_state, --expand, --n_heads, --num_loops, --block_size

Run ``python scripts/init_model.py --help`` for full option list.
"""

import sys
import os
import argparse
import json
from safetensors.torch import save_file

# Allow running from project root or scripts/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from model.vortex import VortexConfig, VortexForCausalLM


# ── Size presets ───────────────────────────────────────────────────────────────

PRESETS = {
    "nano": dict(
        d_model=224,
        n_layer=4,
        d_state=32,
        expand=2,
        n_heads=4,
        dt_rank="auto",
        num_loops=2,
        block_size=1024,
        vocab_size=32000,
    ),
    "small": dict(
        d_model=384,
        n_layer=6,
        d_state=48,
        expand=2,
        n_heads=6,
        dt_rank="auto",
        num_loops=2,
        block_size=1024,
        vocab_size=32000,
    ),
    "medium": dict(
        d_model=512,
        n_layer=8,
        d_state=64,
        expand=2,
        n_heads=8,
        dt_rank="auto",
        num_loops=2,
        block_size=1024,
        vocab_size=32000,
    ),
    "large": dict(
        d_model=768,
        n_layer=12,
        d_state=96,
        expand=2,
        n_heads=12,
        dt_rank="auto",
        num_loops=2,
        block_size=2048,
        vocab_size=32000,
    ),
}


def parse_args():
    p = argparse.ArgumentParser(
        description="Initialize a blank Vortex model and save to disk."
    )
    p.add_argument("--output", default="checkpoints/vortex_v1_small",
                   help="Output directory for model files.")
    p.add_argument("--size", choices=["nano", "small", "medium", "large", "custom"], default="small",
                   help="Model size preset, or 'custom' to specify dimensions manually.")
    p.add_argument("--d_model",   type=int,   default=384)
    p.add_argument("--n_layer",   type=int,   default=6)
    p.add_argument("--d_state",   type=int,   default=48)
    p.add_argument("--expand",    type=int,   default=2)
    p.add_argument("--n_heads",   type=int,   default=6)
    p.add_argument("--num_loops", type=int,   default=2)
    p.add_argument("--block_size",type=int,   default=1024)
    p.add_argument("--vocab_size",type=int,   default=32000)
    p.add_argument("--no_swap",   action="store_true", help="Disable cross-track swap.")
    p.add_argument("--no_triton", action="store_true", help="Disable Triton kernel.")
    p.add_argument("--tie_embeddings", action="store_true", default=True)
    return p.parse_args()


def main():
    args = parse_args()

    if args.size in PRESETS:
        cfg_kwargs = PRESETS[args.size].copy()
    else:
        cfg_kwargs = dict(
            d_model=args.d_model,
            n_layer=args.n_layer,
            d_state=args.d_state,
            expand=args.expand,
            n_heads=args.n_heads,
            dt_rank="auto",
            num_loops=args.num_loops,
            block_size=args.block_size,
            vocab_size=args.vocab_size,
        )

    cfg_kwargs["use_swap"]        = not args.no_swap
    cfg_kwargs["use_triton"]      = not args.no_triton
    cfg_kwargs["tie_embeddings"]  = args.tie_embeddings

    config = VortexConfig(**cfg_kwargs)

    print("=" * 60)
    print("Vortex Model Initialisation")
    print("=" * 60)
    print(config)

    counts = config.count_parameters()
    print("\nParameter estimate:")
    for k, v in counts.items():
        print(f"  {k:50s}: {v:,}" if isinstance(v, int) else f"  {k:50s}: {v}")

    # Build model
    print("\nBuilding model...")
    model = VortexForCausalLM(config)

    actual_params = sum(p.numel() for p in model.parameters())
    actual_unique = sum(p.numel() for p in set(model.parameters()))
    print(f"  Actual total parameters (with shared): {actual_params:,}")
    print(f"  Unique parameters:                     {actual_unique:,}  (~{actual_unique/1e6:.2f}M)")

    # Save
    #os.makedirs(args.output, exist_ok=True)
    #print(f"\nSaving to: {args.output}")
    #model.save_pretrained(args.output, safe_serialization=True)
    #config.save_pretrained(args.output)
    os.makedirs(args.output, exist_ok=True)
    print(f"\nSaving to: {args.output}")
    
    # Save weights manually via safetensors, de-duplicating the tied pair
    state_dict = model.state_dict()
    # Remove the lm_head duplicate — it points to the same tensor as the embedding
    if config.tie_embeddings and "lm_head.weight" in state_dict:
        del state_dict["lm_head.weight"]
    save_file(state_dict, os.path.join(args.output, "model.safetensors"))

    # Save config
    config_dict = config.to_dict()
    with open(os.path.join(args.output, "config.json"), "w") as f:
        json.dump(config_dict, f, indent=2)

    # Write a human-readable summary
    summary = {
        "architecture": "VortexForCausalLM",
        "preset": args.size,
        "config": config.to_dict(),
        "actual_params": actual_params,
        "unique_params": actual_unique,
        "unique_params_M": round(actual_unique / 1e6, 2),
    }
    with open(os.path.join(args.output, "model_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n✓ Model saved to {args.output}")
    print(f"  Files: config.json, model.safetensors, model_summary.json")


if __name__ == "__main__":
    main()
