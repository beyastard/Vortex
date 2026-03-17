"""
tokenize_data.py
================
Tokenize one or more raw text files into packed uint32 numpy arrays (.bin).

These .bin files are memory-mapped at training time for fast, low-overhead
data loading without loading the full corpus into RAM.

Usage
-----
    # Tokenize a single file
    python scripts/tokenize_data.py --input data/raw_text/train.txt --output data/train.bin

    # Tokenize all .txt files in a directory
    python scripts/tokenize_data.py --input data/raw_text/ --output data/train.bin

    # Custom tokenizer
    python scripts/tokenize_data.py --input data/raw_text/train.txt \\
        --output data/train.bin --tokenizer gpt2

Options
-------
    --input       Path to a .txt file or directory of .txt files.
    --output      Path to the output .bin file.
    --tokenizer   HuggingFace tokenizer name/path (default: amd/AMD-Llama-135m
                  falls back to gpt2 if not available without auth).
    --block_size  Sequence length to use for progress reporting (default: 1024).
    --num_proc    Number of parallel tokenization workers (default: 4).
    --add_eos     Add EOS token between documents (default: True).
    --val_split   Fraction of data to hold out for validation when a single
                  input file is given (0.0 = no split). Output is then
                  --output for train and --output with _val suffix for val.
"""

import sys
import os
import argparse
import glob
import numpy as np
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def parse_args():
    p = argparse.ArgumentParser(description="Tokenize text corpus to .bin files.")
    p.add_argument("--input",      required=True,
                   help="Input .txt file or directory of .txt files.")
    p.add_argument("--output",     required=True,
                   help="Output .bin file path (e.g. data/train.bin).")
    p.add_argument("--tokenizer",  default="amd/AMD-Llama-135m",
                   help="HuggingFace tokenizer identifier.")
    p.add_argument("--block_size", type=int, default=1024,
                   help="Block size used for progress reporting.")
    p.add_argument("--num_proc",   type=int, default=4,
                   help="Tokenizer parallelism workers.")
    p.add_argument("--add_eos",    action="store_true", default=True,
                   help="Insert EOS token between documents.")
    p.add_argument("--no_add_eos", dest="add_eos", action="store_false")
    p.add_argument("--val_split",  type=float, default=0.0,
                   help="Fraction to hold out as validation (0.0 = disabled).")
    p.add_argument("--chunk_size", type=int, default=100_000,
                   help="Lines read per chunk for low-memory processing.")
    return p.parse_args()


def load_tokenizer(name: str):
    """Load tokenizer, with fallback to gpt2 if the requested one needs auth."""
    from transformers import AutoTokenizer
    try:
        tok = AutoTokenizer.from_pretrained(name)
        print(f"Loaded tokenizer: {name}  (vocab_size={tok.vocab_size})")
        return tok
    except Exception as e:
        print(f"Could not load '{name}': {e}")
        print("Falling back to gpt2 tokenizer.")
        tok = AutoTokenizer.from_pretrained("gpt2")
        print(f"Loaded tokenizer: gpt2  (vocab_size={tok.vocab_size})")
        return tok


def collect_files(input_path: str):
    """Return list of .txt file paths from input (file or directory)."""
    p = Path(input_path)
    if p.is_file():
        return [str(p)]
    elif p.is_dir():
        files = sorted(glob.glob(str(p / "*.txt")))
        if not files:
            raise FileNotFoundError(f"No .txt files found in {input_path}")
        return files
    else:
        raise FileNotFoundError(f"Input not found: {input_path}")


def tokenize_file(filepath: str, tokenizer, add_eos: bool, chunk_size: int) -> np.ndarray:
    """Tokenize a single text file, returning a 1-D uint32 array."""
    eos_id = tokenizer.eos_token_id or 0

    all_tokens = []
    total_lines = 0
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        chunk = []
        for line in tqdm(f, desc=f"  Reading {Path(filepath).name}", unit="line",
                         dynamic_ncols=True):
            chunk.append(line.rstrip("\n"))
            if len(chunk) >= chunk_size:
                text = "\n".join(chunk)
                ids = tokenizer.encode(text, add_special_tokens=False)
                all_tokens.extend(ids)
                if add_eos:
                    all_tokens.append(eos_id)
                total_lines += len(chunk)
                chunk = []
        if chunk:
            text = "\n".join(chunk)
            ids = tokenizer.encode(text, add_special_tokens=False)
            all_tokens.extend(ids)
            if add_eos:
                all_tokens.append(eos_id)
            total_lines += len(chunk)

    arr = np.array(all_tokens, dtype=np.uint32)
    print(f"  → {total_lines:,} lines, {len(arr):,} tokens from {Path(filepath).name}")
    return arr


def write_bin(tokens: np.ndarray, path: str):
    """Write uint32 token array to binary file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tokens.tofile(path)
    size_mb = tokens.nbytes / 1024 / 1024
    print(f"  Wrote {len(tokens):,} tokens ({size_mb:.1f} MB) → {path}")


def main():
    args = parse_args()

    tokenizer = load_tokenizer(args.tokenizer)
    files = collect_files(args.input)
    print(f"\nFound {len(files)} input file(s).")

    all_tokens_list = []
    for fp in files:
        arr = tokenize_file(fp, tokenizer, args.add_eos, args.chunk_size)
        all_tokens_list.append(arr)

    all_tokens = np.concatenate(all_tokens_list)
    print(f"\nTotal tokens: {len(all_tokens):,}")
    print(f"Estimated sequences of length {args.block_size}: "
          f"{len(all_tokens) // args.block_size:,}")

    if args.val_split > 0.0 and args.val_split < 1.0:
        split_idx = int(len(all_tokens) * (1.0 - args.val_split))
        train_tokens = all_tokens[:split_idx]
        val_tokens   = all_tokens[split_idx:]

        train_path = args.output
        val_path   = str(Path(args.output).with_name(
                         Path(args.output).stem + "_val" + Path(args.output).suffix))

        print(f"\nSplitting: train={len(train_tokens):,}  val={len(val_tokens):,} tokens")
        write_bin(train_tokens, train_path)
        write_bin(val_tokens,   val_path)
    else:
        write_bin(all_tokens, args.output)

    print("\n✓ Tokenization complete.")


if __name__ == "__main__":
    main()
