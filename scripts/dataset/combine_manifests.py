"""
combine_manifests.py
====================
Combine multiple Arrow dataset manifests from tokenize_universal.py
into a single manifest for joint training across all datasets.

The combined manifest simply pools all train and val shard paths from
each individual manifest. The ArrowDataset loader in train.py uses
weighted random shard sampling, so larger datasets automatically
contribute more to each training batch proportionally.

Author: Bryan K Reinhart
License: AGPL-3.0

Usage
-----
    python scripts/dataset/combine_manifests.py \\
        --input_dirs data/arrow/alt_fantasy data/arrow/alt_pantheon ... \\
        --output_dir data/arrow/mickume_combined
"""

import sys
import os
import argparse
import json
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Combine Arrow dataset manifests.")
    p.add_argument("--input_dirs", nargs="+", required=True,
                   help="Directories containing manifest.json files.")
    p.add_argument("--output_dir", required=True,
                   help="Output directory for the combined manifest.")
    return p.parse_args()


def main():
    args = parse_args()

    all_train_shards = []
    all_val_shards   = []
    total_blocks     = 0
    total_tokens     = 0
    total_docs       = 0
    tokenizer        = None
    block_size       = None

    print(f"\nCombining {len(args.input_dirs)} manifests...")
    print(f"{'='*55}")

    for d in args.input_dirs:
        manifest_path = os.path.join(d, "manifest.json")
        if not os.path.exists(manifest_path):
            print(f"  [WARN] No manifest.json in {d} — skipping.")
            continue

        with open(manifest_path) as f:
            manifest = json.load(f)

        name         = Path(d).name
        train_shards = manifest.get("train_shards", [])
        val_shards   = manifest.get("val_shards",   [])
        blocks       = manifest.get("total_blocks",  0)
        tokens       = manifest.get("total_tokens",  0)
        docs         = manifest.get("total_docs",    0)

        # Validate shard files exist
        train_ok = [s for s in train_shards if os.path.exists(s)]
        val_ok   = [s for s in val_shards   if os.path.exists(s)]
        missing  = len(train_shards) - len(train_ok) + len(val_shards) - len(val_ok)
        if missing:
            print(f"  [WARN] {name}: {missing} shard file(s) missing — skipping them.")

        all_train_shards.extend(train_ok)
        all_val_shards.extend(val_ok)
        total_blocks += blocks
        total_tokens += tokens
        total_docs   += docs

        if tokenizer is None:
            tokenizer  = manifest.get("tokenizer")
            block_size = manifest.get("block_size")

        print(f"  {name:<25}  "
              f"train={len(train_ok):3d} shards  "
              f"val={len(val_ok):2d} shards  "
              f"~{tokens/1e6:.0f}M tokens")

    print(f"{'='*55}")
    print(f"  {'TOTAL':<25}  "
          f"train={len(all_train_shards):3d} shards  "
          f"val={len(all_val_shards):2d} shards  "
          f"~{total_tokens/1e6:.0f}M tokens")

    # Write combined manifest
    os.makedirs(args.output_dir, exist_ok=True)
    combined = {
        "format":        "arrow",
        "tokenizer":     tokenizer,
        "block_size":    block_size,
        "total_blocks":  total_blocks,
        "total_tokens":  total_tokens,
        "total_docs":    total_docs,
        "train_shards":  all_train_shards,
        "val_shards":    all_val_shards,
        "source":        "combined",
        "input_dirs":    args.input_dirs,
    }
    out_path = os.path.join(args.output_dir, "manifest.json")
    with open(out_path, "w") as f:
        json.dump(combined, f, indent=2)

    print(f"\n  ✓ Combined manifest → {out_path}")
    print(f"\n  To train on the full combined dataset:")
    print(f"  python scripts\\train.py \\")
    print(f"      --data_format arrow \\")
    print(f"      --arrow_manifest {out_path}")


if __name__ == "__main__":
    main()
