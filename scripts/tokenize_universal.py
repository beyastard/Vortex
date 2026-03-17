"""
tokenize_universal.py
=====================
Universal streaming tokenizer for Vortex training data preparation.

Supports all major dataset formats and produces Arrow IPC files as the
primary output format — the most efficient format for large-scale NLP
training due to zero-copy memory mapping, columnar compression, and
native HuggingFace datasets integration.

Also supports .bin (uint32 numpy) output for compatibility with the
existing BinaryDataset loader in train.py.

Author: Bryan K Reinhart
License: AGPL-3.0

Supported Input Formats
-----------------------
  .parquet          Apache Parquet (columnar, compressed)
  .jsonl / .json    JSON Lines or JSON array
  .csv / .tsv       Comma / tab separated values
  .txt              Plain text (one document per line or full file)
  .arrow            Apache Arrow IPC (pass-through re-tokenize)
  HuggingFace Hub   Any dataset loadable via datasets.load_dataset()

Output Formats
--------------
  arrow (default)   Arrow IPC dataset shard files — memory-mappable,
                    directly loadable with datasets.Dataset.from_file().
                    Stores input_ids as int32 sequences, one row per
                    packed block of block_size tokens. Zero padding never
                    occurs — documents are packed end-to-end with EOS
                    tokens between them (next-token-prediction optimal).

  bin               Flat uint32 numpy binary file — compatible with the
                    BinaryDataset loader in scripts/train.py.

Field Extraction
----------------
The --field_map argument controls how text is extracted from each record.
It accepts a JSON string describing extraction rules. Examples:

  Simple field rename:
    --field_map '{"type": "field", "field": "texts"}'

  Nested field (dot notation):
    --field_map '{"type": "field", "field": "meta.text"}'

  Conversation/chat format (list of message dicts):
    --field_map '{
      "type": "conversation",
      "field": "conversations",
      "role_key": "from",
      "content_key": "value",
      "roles": {"human": "User", "gpt": "Assistant"},
      "separator": "\\n\\n"
    }'

  ShareGPT / OpenAI messages format:
    --field_map '{
      "type": "conversation",
      "field": "messages",
      "role_key": "role",
      "content_key": "content",
      "roles": {"user": "User", "assistant": "Assistant", "system": "System"},
      "separator": "\\n\\n"
    }'

  Multiple fields joined together:
    --field_map '{
      "type": "join",
      "fields": ["title", "abstract", "body"],
      "separator": "\\n\\n"
    }'

  Field with score filter (keep docs where avg_score < 0.5):
    --field_map '{
      "type": "field",
      "field": "texts",
      "filter": {"field": "avg_score", "op": "<", "value": 0.5}
    }'

  Field with multiple filters (AND logic):
    --field_map '{
      "type": "field",
      "field": "text",
      "filters": [
        {"field": "avg_score", "op": "<", "value": 0.5},
        {"field": "num_sents", "op": ">=", "value": 3}
      ]
    }'

  Conditional: use field A if present, else field B:
    --field_map '{
      "type": "coalesce",
      "fields": ["text", "content", "body", "description"]
    }'

  Template: format multiple fields into a string:
    --field_map '{
      "type": "template",
      "template": "Title: {title}\\n\\nAbstract: {abstract}",
      "fields": ["title", "abstract"]
    }'

Dataset-specific presets (use --preset instead of --field_map):
  arxiv_tex         KiteFishAI/arxiv-tex-corpus-full  (field: text)
  gpt_oss_sft       Pinkstack/gpt_oss_sft             (conversations)
  pile_detoxify     tomekkorbak/pile-detoxify-test     (texts, filter avg_score<0.5)
  tinystories       roneneldan/TinyStories             (field: text)
  openwebtext       openwebtext                        (field: text)
  sharegpt          ShareGPT-style datasets            (conversations/messages)
  alpaca            tatsu-lab/alpaca style             (instruction+input+output)
  dolly             databricks/databricks-dolly-15k    (instruction+context+response)

"""

from __future__ import annotations

import sys
import os
import argparse
import glob
import json
import math
import random
import hashlib
import time
import re
from pathlib import Path
from typing import Iterator, Optional, Any

import numpy as np
from tqdm import tqdm


# ──────────────────────────────────────────────────────────────────────────────
# Dataset presets
# ──────────────────────────────────────────────────────────────────────────────

PRESETS: dict[str, dict] = {
    "arxiv_tex": {
        "type": "field",
        "field": "text",
    },
    "gpt_oss_sft": {
        "type": "conversation",
        "field": "conversations",
        "role_key": "from",
        "content_key": "value",
        "roles": {"human": "User", "gpt": "Assistant", "system": "System"},
        "separator": "\n\n",
        "include_roles": True,
    },
    "pile_detoxify": {
        "type": "field",
        "field": "texts",
        "filters": [
            {"field": "avg_score", "op": "<", "value": 0.5},
        ],
    },
    "tinystories": {
        "type": "field",
        "field": "text",
    },
    "openwebtext": {
        "type": "field",
        "field": "text",
    },
    "sharegpt": {
        "type": "conversation",
        "field": "conversations",
        "role_key": "from",
        "content_key": "value",
        "roles": {"human": "User", "gpt": "Assistant", "system": "System"},
        "separator": "\n\n",
        "include_roles": True,
    },
    "openai_messages": {
        "type": "conversation",
        "field": "messages",
        "role_key": "role",
        "content_key": "content",
        "roles": {
            "user": "User",
            "assistant": "Assistant",
            "system": "System",
        },
        "separator": "\n\n",
        "include_roles": True,
    },
    "alpaca": {
        "type": "template",
        "template": (
            "### Instruction:\n{instruction}"
            "{input_section}"
            "\n\n### Response:\n{output}"
        ),
        "fields": ["instruction", "input", "output"],
        "_alpaca_mode": True,  # special handling for optional input field
    },
    "dolly": {
        "type": "template",
        "template": (
            "### Instruction:\n{instruction}"
            "{context_section}"
            "\n\n### Response:\n{response}"
        ),
        "fields": ["instruction", "context", "response"],
        "_dolly_mode": True,
    },
}


# ──────────────────────────────────────────────────────────────────────────────
# Lazy imports
# ──────────────────────────────────────────────────────────────────────────────

def _import_pyarrow():
    try:
        import pyarrow as pa
        import pyarrow.ipc as ipc
        import pyarrow.compute as pc
        return pa, ipc, pc
    except ImportError:
        raise ImportError(
            "pyarrow is required for Arrow output.\n"
            "Install with:  pip install pyarrow"
        )

def _import_datasets():
    try:
        import datasets
        return datasets
    except ImportError:
        raise ImportError(
            "datasets is required for HuggingFace Hub and Parquet support.\n"
            "Install with:  pip install datasets"
        )

def _import_pandas():
    try:
        import pandas as pd
        return pd
    except ImportError:
        raise ImportError(
            "pandas is required for CSV/TSV support.\n"
            "Install with:  pip install pandas"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Field extraction engine
# ──────────────────────────────────────────────────────────────────────────────

class FieldExtractor:
    """
    Extracts a text string from a dataset record according to a field map
    specification. Handles all field types, conversation formats, filters,
    templates, and coalesce logic.

    The extractor is initialised once and called for each record in the
    streaming pipeline — it is designed to be fast and allocation-efficient.
    """

    def __init__(self, field_map: dict):
        self.spec    = field_map
        self.ftype   = field_map.get("type", "field")
        self._filter_cache: list = self._compile_filters(field_map)

    # ── Filter compilation ────────────────────────────────────────────────────

    def _compile_filters(self, spec: dict) -> list:
        """
        Compile filter specs into callable (field, op_fn) pairs.
        Supports single 'filter' or list 'filters'.
        """
        filters = []
        single  = spec.get("filter")
        multi   = spec.get("filters", [])

        all_specs = ([single] if single else []) + list(multi)
        for f in all_specs:
            field = f["field"]
            op    = f["op"]
            val   = f["value"]
            if op == "<":
                fn = lambda x, v=val: x < v
            elif op == "<=":
                fn = lambda x, v=val: x <= v
            elif op == ">":
                fn = lambda x, v=val: x > v
            elif op == ">=":
                fn = lambda x, v=val: x >= v
            elif op == "==":
                fn = lambda x, v=val: x == v
            elif op == "!=":
                fn = lambda x, v=val: x != v
            elif op == "in":
                fn = lambda x, v=val: x in v
            elif op == "not_in":
                fn = lambda x, v=val: x not in v
            elif op == "contains":
                fn = lambda x, v=val: v in str(x)
            else:
                raise ValueError(f"Unknown filter op: '{op}'. "
                                  f"Supported: <, <=, >, >=, ==, !=, in, not_in, contains")
            filters.append((field, fn))
        return filters

    def _passes_filters(self, record: dict) -> bool:
        """Return True if the record passes all filters."""
        for field, fn in self._filter_cache:
            # Support dot-notation for nested filter fields
            val = self._get_nested(record, field)
            if val is None:
                return False
            try:
                if not fn(val):
                    return False
            except (TypeError, ValueError):
                return False
        return True

    # ── Field access ──────────────────────────────────────────────────────────

    @staticmethod
    def _get_nested(record: dict, field: str) -> Any:
        """
        Get a value from a record using dot notation.
        e.g. 'meta.text' → record['meta']['text']
        Returns None if any key is missing.
        """
        parts = field.split(".")
        val   = record
        for part in parts:
            if isinstance(val, dict):
                val = val.get(part)
            elif isinstance(val, (list, tuple)):
                try:
                    val = val[int(part)]
                except (ValueError, IndexError):
                    return None
            else:
                return None
            if val is None:
                return None
        return val

    # ── Extraction methods ────────────────────────────────────────────────────

    def _extract_field(self, record: dict) -> Optional[str]:
        field = self.spec.get("field", "text")
        val   = self._get_nested(record, field)
        if val is None:
            return None
        if isinstance(val, list):
            # List of strings — join them
            return "\n".join(str(v) for v in val if v)
        return str(val)

    def _extract_conversation(self, record: dict) -> Optional[str]:
        """
        Extract and format a conversation from a list of message dicts.

        Handles multiple formats:
          ShareGPT:  [{"from": "human", "value": "..."}, ...]
          OpenAI:    [{"role": "user", "content": "..."}, ...]
          Simple:    [{"role": "...", "text": "..."}, ...]
        """
        field       = self.spec.get("field", "conversations")
        role_key    = self.spec.get("role_key", "from")
        content_key = self.spec.get("content_key", "value")
        roles       = self.spec.get("roles", {})
        separator   = self.spec.get("separator", "\n\n")
        include_roles = self.spec.get("include_roles", True)
        skip_roles  = set(self.spec.get("skip_roles", []))

        messages = self._get_nested(record, field)
        if not isinstance(messages, list) or len(messages) == 0:
            return None

        parts = []
        for msg in messages:
            if not isinstance(msg, dict):
                continue

            # Get role
            role = msg.get(role_key, "")
            if role in skip_roles:
                continue

            # Get content — try multiple common key names
            content = None
            for key in [content_key, "value", "content", "text", "message"]:
                if key in msg and msg[key]:
                    content = str(msg[key]).strip()
                    break

            if not content:
                continue

            # Format role label
            role_label = roles.get(role, role.capitalize() if role else "")

            if include_roles and role_label:
                parts.append(f"{role_label}: {content}")
            else:
                parts.append(content)

        if not parts:
            return None

        return separator.join(parts)

    def _extract_join(self, record: dict) -> Optional[str]:
        """Join multiple fields together with a separator."""
        fields    = self.spec.get("fields", [])
        separator = self.spec.get("separator", "\n\n")
        labels    = self.spec.get("labels", {})  # optional field labels

        parts = []
        for field in fields:
            val = self._get_nested(record, field)
            if val is None or str(val).strip() == "":
                if self.spec.get("require_all", False):
                    return None
                continue
            text = str(val).strip()
            label = labels.get(field)
            if label:
                parts.append(f"{label}: {text}")
            else:
                parts.append(text)

        return separator.join(parts) if parts else None

    def _extract_coalesce(self, record: dict) -> Optional[str]:
        """Return the first non-empty field value from a list of candidates."""
        for field in self.spec.get("fields", []):
            val = self._get_nested(record, field)
            if val is not None and str(val).strip():
                return str(val).strip()
        return None

    def _extract_template(self, record: dict) -> Optional[str]:
        """Format fields into a template string."""
        template = self.spec.get("template", "")
        fields   = self.spec.get("fields", [])

        # Special handling for Alpaca format (optional 'input' field)
        if self.spec.get("_alpaca_mode"):
            instruction = str(self._get_nested(record, "instruction") or "").strip()
            inp         = str(self._get_nested(record, "input") or "").strip()
            output      = str(self._get_nested(record, "output") or "").strip()
            if not instruction or not output:
                return None
            input_section = f"\n\n### Input:\n{inp}" if inp else ""
            try:
                return template.format(
                    instruction=instruction,
                    input_section=input_section,
                    output=output,
                )
            except KeyError:
                return None

        # Special handling for Dolly format (optional 'context' field)
        if self.spec.get("_dolly_mode"):
            instruction = str(self._get_nested(record, "instruction") or "").strip()
            context     = str(self._get_nested(record, "context") or "").strip()
            response    = str(self._get_nested(record, "response") or "").strip()
            if not instruction or not response:
                return None
            context_section = f"\n\n### Context:\n{context}" if context else ""
            try:
                return template.format(
                    instruction=instruction,
                    context_section=context_section,
                    response=response,
                )
            except KeyError:
                return None

        # General template
        values = {}
        for field in fields:
            val = self._get_nested(record, field)
            values[field.replace(".", "_")] = str(val).strip() if val else ""

        # Also make raw field names available
        for field in fields:
            val = self._get_nested(record, field)
            key = field.split(".")[-1]  # last segment of dot notation
            values[key] = str(val).strip() if val else ""

        try:
            return template.format(**values) or None
        except KeyError as e:
            return None

    # ── Public interface ──────────────────────────────────────────────────────

    def extract(self, record: dict) -> Optional[str]:
        """
        Extract text from a record. Returns None if:
          - Required fields are missing
          - Any filter fails
          - The result is empty
        """
        # Check filters first (fast path — skip tokenisation of filtered docs)
        if self._filter_cache and not self._passes_filters(record):
            return None

        # Dispatch to extraction method
        if self.ftype == "field":
            text = self._extract_field(record)
        elif self.ftype == "conversation":
            text = self._extract_conversation(record)
        elif self.ftype == "join":
            text = self._extract_join(record)
        elif self.ftype == "coalesce":
            text = self._extract_coalesce(record)
        elif self.ftype == "template":
            text = self._extract_template(record)
        else:
            raise ValueError(
                f"Unknown field map type: '{self.ftype}'. "
                f"Supported: field, conversation, join, coalesce, template"
            )

        if text is None or not text.strip():
            return None
        return text.strip()

    @classmethod
    def from_args(cls, args) -> "FieldExtractor":
        """Build a FieldExtractor from parsed CLI args."""
        if hasattr(args, "preset") and args.preset:
            if args.preset not in PRESETS:
                raise ValueError(
                    f"Unknown preset '{args.preset}'. "
                    f"Available: {', '.join(PRESETS)}"
                )
            spec = PRESETS[args.preset].copy()
            print(f"  Using preset: {args.preset}")
            print(f"  Field map:    {json.dumps(spec, indent=2)}")
            return cls(spec)

        if hasattr(args, "field_map") and args.field_map:
            try:
                spec = json.loads(args.field_map)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid --field_map JSON: {e}")
            print(f"  Field map: {json.dumps(spec, indent=2)}")
            return cls(spec)

        # Default: simple field extraction using --text_field
        spec = {"type": "field", "field": args.text_field}
        return cls(spec)

    def describe(self) -> str:
        """Human-readable description of this extractor."""
        t = self.ftype
        if t == "field":
            desc = f"field '{self.spec.get('field', 'text')}'"
        elif t == "conversation":
            desc = (f"conversation from '{self.spec.get('field')}' "
                    f"(role_key='{self.spec.get('role_key', 'from')}', "
                    f"content_key='{self.spec.get('content_key', 'value')}')")
        elif t == "join":
            desc = f"join {self.spec.get('fields', [])}"
        elif t == "coalesce":
            desc = f"coalesce {self.spec.get('fields', [])}"
        elif t == "template":
            desc = f"template '{self.spec.get('template', '')[:60]}...'"
        else:
            desc = str(self.spec)

        if self._filter_cache:
            filters = self.spec.get("filters", [])
            if self.spec.get("filter"):
                filters = [self.spec["filter"]] + filters
            filter_str = ", ".join(
                f"{f['field']} {f['op']} {f['value']}" for f in filters
            )
            desc += f"  [filter: {filter_str}]"

        return desc


# ──────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Universal streaming tokenizer for Vortex.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Input
    inp = p.add_mutually_exclusive_group(required=True)
    inp.add_argument("--input",      default=None,
                     help="Local file or directory.")
    inp.add_argument("--hf_dataset", default=None,
                     help="HuggingFace dataset name.")

    p.add_argument("--file_type",   default=None,
                   choices=["parquet", "jsonl", "json", "csv", "tsv", "txt", "arrow"],
                   help="Force file type detection.")
    p.add_argument("--hf_config",   default=None,
                   help="HuggingFace dataset config/subset.")
    p.add_argument("--hf_split",    default="train",
                   help="HuggingFace dataset split (default: train).")

    # Field extraction — three ways to specify (mutually exclusive)
    field_grp = p.add_mutually_exclusive_group()
    field_grp.add_argument("--preset", default=None,
                            choices=list(PRESETS),
                            help="Use a built-in dataset preset.")
    field_grp.add_argument("--field_map", default=None,
                            help="JSON field map specification (see docstring).")
    field_grp.add_argument("--text_field", default="text",
                            help="Simple field name for text (default: text).")

    # CSV options
    p.add_argument("--csv_delimiter", default=",")
    p.add_argument("--txt_mode",      default="paragraphs",
                   choices=["lines", "paragraphs", "file"])

    # Tokenizer
    p.add_argument("--tokenizer",       default="amd/AMD-Llama-135m",
                   help="HuggingFace tokenizer (default: amd/AMD-Llama-135m).")
    p.add_argument("--tokenizer_batch", type=int, default=1000,
                   help="Documents per tokenizer batch (default: 1000).")

    # Output
    p.add_argument("--output_dir",  required=True)
    p.add_argument("--format",      default="arrow",
                   choices=["arrow", "bin"],
                   help="Output format (default: arrow).")
    p.add_argument("--block_size",  type=int, default=1024)
    p.add_argument("--shard_size",  type=int, default=50_000_000,
                   help="Tokens per Arrow shard (default: 50M).")
    p.add_argument("--val_split",   type=float, default=0.05,
                   help="Validation fraction (default: 0.05).")
    p.add_argument("--val_shards",  type=int,   default=0,
                   help="Exact number of val shards (overrides --val_split).")
    p.add_argument("--compression", default="zstd",
                   choices=["lz4", "zstd", "none"])
    p.add_argument("--no_pack",     action="store_true",
                   help="Pad to block_size instead of packing documents.")

    # Filtering
    p.add_argument("--min_doc_chars", type=int, default=64)
    p.add_argument("--max_doc_chars", type=int, default=0)
    p.add_argument("--dedupe_prefix", type=int, default=0)

    # Misc
    p.add_argument("--seed",    type=int, default=42)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--resume",  action="store_true")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--show_samples", type=int, default=0,
                   help="Print N sample extracted texts before tokenizing.")

    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Tokenizer loader
# ──────────────────────────────────────────────────────────────────────────────

def load_tokenizer(name: str):
    from transformers import AutoTokenizer
    try:
        tok = AutoTokenizer.from_pretrained(name)
        print(f"  Tokenizer : {name}")
        print(f"  Vocab size: {tok.vocab_size:,}")
        return tok
    except Exception as e:
        print(f"  [WARN] Could not load '{name}': {e}")
        print("  Falling back to gpt2 tokenizer.")
        tok = AutoTokenizer.from_pretrained("gpt2")
        print(f"  Tokenizer : gpt2  (vocab_size={tok.vocab_size})")
        return tok


# ──────────────────────────────────────────────────────────────────────────────
# Record iterators — yield raw dicts from each format
# ──────────────────────────────────────────────────────────────────────────────

def iter_records_txt(path: str, mode: str) -> Iterator[dict]:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        if mode == "file":
            yield {"text": f.read()}
        elif mode == "lines":
            for line in f:
                yield {"text": line.rstrip("\n")}
        elif mode == "paragraphs":
            buf = []
            for line in f:
                stripped = line.rstrip("\n")
                if stripped == "":
                    if buf:
                        yield {"text": "\n".join(buf)}
                    buf = []
                else:
                    buf.append(stripped)
            if buf:
                yield {"text": "\n".join(buf)}


def iter_records_jsonl(path: str) -> Iterator[dict]:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def iter_records_json(path: str) -> Iterator[dict]:
    """Try JSONL first, then fall back to JSON array via ijson."""
    # Check if it's a JSON array or JSONL by peeking at first char
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        first = f.read(1)

    if first == "[":
        # JSON array — use ijson for streaming
        try:
            import ijson
            with open(path, "rb") as f:
                for item in ijson.items(f, "item"):
                    if isinstance(item, dict):
                        yield item
        except ImportError:
            # Fall back to loading the whole file if ijson not available
            print("  [WARN] ijson not installed — loading full JSON into memory.")
            print("         Install with: pip install ijson")
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        yield item
    else:
        # JSONL
        yield from iter_records_jsonl(path)


def iter_records_csv(path: str, delimiter: str) -> Iterator[dict]:
    pd = _import_pandas()
    reader = pd.read_csv(
        path,
        delimiter=delimiter,
        chunksize=10_000,
        on_bad_lines="skip",
        engine="python",
        encoding="utf-8",
        encoding_errors="replace",
    )
    for chunk in reader:
        for _, row in chunk.iterrows():
            yield row.to_dict()


def iter_records_parquet(path: str) -> Iterator[dict]:
    pa, ipc, pc = _import_pyarrow()
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=5_000):
        # Convert to list of dicts without loading full file
        schema_names = batch.schema.names
        rows = batch.to_pydict()
        n = batch.num_rows
        for i in range(n):
            yield {col: rows[col][i] for col in schema_names}


def iter_records_arrow(path: str) -> Iterator[dict]:
    pa, ipc, pc = _import_pyarrow()
    with pa.memory_map(path, "r") as source:
        reader = ipc.open_file(source)
        for i in range(reader.num_record_batches):
            batch = reader.get_batch(i)
            rows  = batch.to_pydict()
            names = batch.schema.names
            for j in range(batch.num_rows):
                yield {col: rows[col][j] for col in names}


def iter_records_hf(dataset_name: str, config: Optional[str],
                    split: str) -> Iterator[dict]:
    datasets = _import_datasets()
    ds = datasets.load_dataset(
        dataset_name,
        config,
        split=split,
        streaming=True,
        trust_remote_code=True,
    )
    yield from ds


# ──────────────────────────────────────────────────────────────────────────────
# File type detection
# ──────────────────────────────────────────────────────────────────────────────

def _detect_file_type(path: str) -> str:
    ext = Path(path).suffix.lower()
    return {
        ".parquet": "parquet",
        ".jsonl":   "jsonl",
        ".json":    "json",
        ".csv":     "csv",
        ".tsv":     "tsv",
        ".txt":     "txt",
        ".text":    "txt",
        ".arrow":   "arrow",
        ".ipc":     "arrow",
    }.get(ext, "txt")


def collect_input_files(input_path: str,
                         forced_type: Optional[str]) -> list[tuple[str, str]]:
    p = Path(input_path)
    if p.is_file():
        return [(str(p), forced_type or _detect_file_type(str(p)))]
    elif p.is_dir():
        ext_map = {
            "parquet": ["*.parquet"],
            "jsonl":   ["*.jsonl"],
            "json":    ["*.json"],
            "csv":     ["*.csv"],
            "tsv":     ["*.tsv"],
            "txt":     ["*.txt", "*.text"],
            "arrow":   ["*.arrow", "*.ipc"],
            None:      ["*.parquet", "*.jsonl", "*.json", "*.csv",
                        "*.tsv", "*.txt", "*.text", "*.arrow"],
        }
        results = []
        for pattern in ext_map.get(forced_type, ext_map[None]):
            for fp in sorted(p.glob(pattern)):
                results.append((str(fp), forced_type or _detect_file_type(str(fp))))
        if not results:
            raise FileNotFoundError(f"No supported files found in {input_path}")
        return results
    else:
        raise FileNotFoundError(f"Input not found: {input_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Master document iterator
# ──────────────────────────────────────────────────────────────────────────────

def make_document_iterator(
    args,
    extractor: FieldExtractor,
    files: Optional[list] = None,
) -> Iterator[str]:
    """
    Unified document iterator. Routes records from any source through
    the FieldExtractor, applies length filtering and optional deduplication,
    and yields clean text strings ready for tokenization.
    """
    seen_hashes = set() if args.dedupe_prefix > 0 else None

    def _process_record(record: dict) -> Optional[str]:
        text = extractor.extract(record)
        if text is None:
            return None

        # Length filter
        if len(text) < args.min_doc_chars:
            return None
        if args.max_doc_chars and len(text) > args.max_doc_chars:
            return None

        # Deduplication
        if seen_hashes is not None:
            prefix = text[:args.dedupe_prefix]
            h = hashlib.md5(prefix.encode("utf-8", errors="replace")).digest()
            if h in seen_hashes:
                return None
            seen_hashes.add(h)

        return text

    if args.hf_dataset:
        for record in iter_records_hf(args.hf_dataset, args.hf_config,
                                       args.hf_split):
            text = _process_record(record)
            if text is not None:
                yield text
        return

    for filepath, ftype in files:
        if ftype == "txt":
            record_iter = iter_records_txt(filepath, args.txt_mode)
        elif ftype == "jsonl":
            record_iter = iter_records_jsonl(filepath)
        elif ftype == "json":
            record_iter = iter_records_json(filepath)
        elif ftype in ("csv", "tsv"):
            delim = "\t" if ftype == "tsv" else args.csv_delimiter
            record_iter = iter_records_csv(filepath, delim)
        elif ftype == "parquet":
            record_iter = iter_records_parquet(filepath)
        elif ftype == "arrow":
            record_iter = iter_records_arrow(filepath)
        else:
            print(f"  [WARN] Skipping unsupported file type '{ftype}': {filepath}")
            continue

        for record in record_iter:
            text = _process_record(record)
            if text is not None:
                yield text


# ──────────────────────────────────────────────────────────────────────────────
# Token packing
# ──────────────────────────────────────────────────────────────────────────────

class TokenPacker:
    def __init__(self, block_size: int, eos_id: int,
                 no_pack: bool = False, pad_id: int = 0):
        self.block_size = block_size
        self.eos_id     = eos_id
        self.no_pack    = no_pack
        self.pad_id     = pad_id
        self._buf: list[int] = []
        self.blocks_emitted  = 0
        self.tokens_seen     = 0
        self.docs_seen       = 0
        self.docs_filtered   = 0

    def feed(self, token_ids: list[int]) -> list[np.ndarray]:
        self.docs_seen   += 1
        self.tokens_seen += len(token_ids) + 1
        completed = []

        if self.no_pack:
            ids = token_ids[:self.block_size]
            if len(ids) < self.block_size:
                ids = ids + [self.pad_id] * (self.block_size - len(ids))
            completed.append(np.array(ids, dtype=np.int32))
            self.blocks_emitted += 1
            return completed

        self._buf.extend(token_ids)
        self._buf.append(self.eos_id)

        while len(self._buf) >= self.block_size + 1:
            block = np.array(self._buf[:self.block_size + 1], dtype=np.int32)
            completed.append(block)
            self._buf = self._buf[self.block_size + 1:]
            self.blocks_emitted += 1

        return completed

    def flush(self) -> Optional[np.ndarray]:
        if len(self._buf) < 2:
            return None
        ids   = self._buf + [self.pad_id] * (self.block_size + 1 - len(self._buf))
        block = np.array(ids[:self.block_size + 1], dtype=np.int32)
        self._buf = []
        self.blocks_emitted += 1
        return block


# ──────────────────────────────────────────────────────────────────────────────
# Arrow shard writer
# ──────────────────────────────────────────────────────────────────────────────

class ArrowShardWriter:
    def __init__(self, output_dir: str, split: str, block_size: int,
                 shard_size_tokens: int, compression: str, resume: bool):
        self.output_dir   = output_dir
        self.split        = split
        self.block_size   = block_size
        self.shard_blocks = max(1, shard_size_tokens // block_size)
        self.compression  = None if compression == "none" else compression
        self.resume       = resume
        self._buf_input: list  = []
        self._buf_labels: list = []
        self._buf_lengths: list = []
        self._shard_idx   = 0
        self._total_blocks = 0
        os.makedirs(output_dir, exist_ok=True)

        if resume:
            existing = sorted(Path(output_dir).glob(f"{split}_shard_*.arrow"))
            if existing:
                try:
                    self._shard_idx = int(existing[-1].stem.split("_")[-1]) + 1
                    print(f"  Resuming {split}: starting at shard {self._shard_idx}")
                except ValueError:
                    pass

        self._pa, self._ipc, _ = _import_pyarrow()

    @property
    def schema(self):
        pa = self._pa
        return pa.schema([
            pa.field("input_ids", pa.list_(pa.int32())),
            pa.field("labels",    pa.list_(pa.int32())),
            pa.field("length",    pa.int32()),
        ])

    def _shard_path(self, idx: int) -> str:
        return os.path.join(self.output_dir, f"{self.split}_shard_{idx:05d}.arrow")

    def write_block(self, block: np.ndarray):
        input_ids = block[:-1].tolist()
        labels    = block[1:].tolist()
        length    = int(np.sum(block[:-1] != 0))
        self._buf_input.append(input_ids)
        self._buf_labels.append(labels)
        self._buf_lengths.append(length)
        self._total_blocks += 1
        if len(self._buf_input) >= self.shard_blocks:
            self._flush_shard()

    def _flush_shard(self):
        if not self._buf_input:
            return
        pa  = self._pa
        ipc = self._ipc
        table = pa.table({
            "input_ids": self._buf_input,
            "labels":    self._buf_labels,
            "length":    self._buf_lengths,
        }, schema=self.schema)
        path = self._shard_path(self._shard_idx)
        opts = ipc.IpcWriteOptions(
            compression=pa.Codec(self.compression)
            if self.compression else None
        )
        with ipc.new_file(path, self.schema, options=opts) as writer:
            writer.write_table(table)
        shard_mb = os.path.getsize(path) / 1024 / 1024
        print(f"  → Shard {self._shard_idx:05d} [{self.split}]: "
              f"{len(self._buf_input):,} blocks  ({shard_mb:.1f} MB)")
        self._buf_input   = []
        self._buf_labels  = []
        self._buf_lengths = []
        self._shard_idx  += 1

    def close(self) -> int:
        self._flush_shard()
        return self._total_blocks

    @property
    def shard_count(self) -> int:
        return self._shard_idx


# ──────────────────────────────────────────────────────────────────────────────
# Binary (.bin) writer
# ──────────────────────────────────────────────────────────────────────────────

class BinWriter:
    def __init__(self, path: str, resume: bool):
        self.path  = path
        mode       = "ab" if resume and os.path.exists(path) else "wb"
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._f    = open(path, mode)
        self._total = 0

    def write_block(self, block: np.ndarray):
        ids = block[:-1].astype(np.uint32)
        self._f.write(ids.tobytes())
        self._total += len(ids)

    def close(self) -> int:
        self._f.close()
        size_mb = os.path.getsize(self.path) / 1024 / 1024
        print(f"  → {self.path}: {self._total:,} tokens ({size_mb:.1f} MB)")
        return self._total


# ──────────────────────────────────────────────────────────────────────────────
# Manifest
# ──────────────────────────────────────────────────────────────────────────────

def write_manifest(output_dir: str, train_shards: list,
                   val_shards: list, args, stats: dict):
    manifest = {
        "format":       args.format,
        "tokenizer":    args.tokenizer,
        "block_size":   args.block_size,
        "vocab_size":   stats.get("vocab_size", 0),
        "total_blocks": stats.get("total_blocks", 0),
        "total_tokens": stats.get("total_tokens", 0),
        "total_docs":   stats.get("total_docs", 0),
        "train_shards": train_shards,
        "val_shards":   val_shards,
        "compression":  args.compression if args.format == "arrow" else "none",
        "created":      time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source":       args.hf_dataset or args.input,
        "preset":       getattr(args, "preset", None),
        "field_map":    getattr(args, "field_map", None),
    }
    path = os.path.join(output_dir, "manifest.json")
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n  Manifest → {path}")
    return manifest


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    print(f"\n{'='*65}")
    print(f"  Vortex Universal Tokenizer")
    print(f"{'='*65}")
    print(f"  Source     : {args.hf_dataset or args.input}")
    print(f"  Output dir : {args.output_dir}")
    print(f"  Format     : {args.format}")
    print(f"  Block size : {args.block_size}")
    if args.format == "arrow":
        print(f"  Shard size : {args.shard_size:,} tokens")
        print(f"  Compression: {args.compression}")
    print()

    # ── Build field extractor ─────────────────────────────────────────────────
    extractor = FieldExtractor.from_args(args)
    print(f"  Extraction : {extractor.describe()}")
    print()

    # ── Load tokenizer ────────────────────────────────────────────────────────
    print("Loading tokenizer...")
    tokenizer = load_tokenizer(args.tokenizer)
    eos_id = tokenizer.eos_token_id or 1
    pad_id = tokenizer.pad_token_id or 0

    # ── Collect files ─────────────────────────────────────────────────────────
    files = None
    if args.input:
        files = collect_input_files(args.input, args.file_type)
        print(f"\nInput files ({len(files)}):")
        for fp, ft in files[:10]:
            size_mb = os.path.getsize(fp) / 1024 / 1024
            print(f"  [{ft:8s}] {Path(fp).name}  ({size_mb:.1f} MB)")
        if len(files) > 10:
            print(f"  ... and {len(files)-10} more")

    # ── Sample preview ────────────────────────────────────────────────────────
    if args.show_samples > 0:
        print(f"\nSample extracted texts (first {args.show_samples}):")
        print("-" * 65)
        count = 0
        for text in make_document_iterator(args, extractor, files):
            print(f"\n[Sample {count+1}]\n{text[:500]}"
                  f"{'...' if len(text) > 500 else ''}")
            count += 1
            if count >= args.show_samples:
                break
        print("-" * 65)
        print("\nProceed with full tokenization? (Ctrl+C to abort)\n")
        # Re-create the iterator for the full run
        # (files is stateless so this is safe)

    # ── Dry run ───────────────────────────────────────────────────────────────
    if args.dry_run:
        print("\nDry run — counting documents...")
        n_docs = 0
        total_chars = 0
        for text in tqdm(make_document_iterator(args, extractor, files),
                         desc="Counting", unit="doc", dynamic_ncols=True):
            n_docs += 1
            total_chars += len(text)
        avg = total_chars / max(n_docs, 1)
        est_tokens = int(total_chars * 0.25)
        est_blocks  = est_tokens // args.block_size
        print(f"\n  Documents     : {n_docs:,}")
        print(f"  Avg doc chars : {avg:,.0f}")
        print(f"  Est tokens    : {est_tokens:,}  (~{est_tokens/1e9:.3f}B)")
        print(f"  Est blocks    : {est_blocks:,}")
        if args.format == "arrow":
            est_shards = max(1, est_tokens // args.shard_size)
            print(f"  Est shards    : {est_shards:,}")
        print(f"\n[DRY RUN complete]")
        return

    # ── Set up writers ────────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)

    if args.format == "arrow":
        stage_dir = os.path.join(args.output_dir, "_staging")
        writer = ArrowShardWriter(
            stage_dir, "all", args.block_size,
            args.shard_size, args.compression, args.resume,
        )
    else:
        train_writer = BinWriter(
            os.path.join(args.output_dir, "train.bin"), args.resume
        )
        val_writer = BinWriter(
            os.path.join(args.output_dir, "val.bin"), args.resume
        )

    packer = TokenPacker(args.block_size, eos_id, args.no_pack, pad_id)
    total_blocks_written = 0

    # ── Streaming tokenization loop ───────────────────────────────────────────
    pbar      = tqdm(desc="Documents", unit="doc", dynamic_ncols=True)
    batch_buf: list[str] = []

    def flush_batch(docs: list[str]):
        nonlocal total_blocks_written
        if not docs:
            return
        enc = tokenizer(
            docs,
            add_special_tokens=False,
            truncation=False,
            padding=False,
            return_attention_mask=False,
        )
        for ids in enc["input_ids"]:
            for block in packer.feed(ids):
                if args.format == "arrow":
                    writer.write_block(block)
                else:
                    if random.random() < args.val_split:
                        val_writer.write_block(block)
                    else:
                        train_writer.write_block(block)
                total_blocks_written += 1

    try:
        for text in make_document_iterator(args, extractor, files):
            batch_buf.append(text)
            pbar.update(1)
            pbar.set_postfix({
                "blocks": f"{total_blocks_written:,}",
                "M_tok":  f"{packer.tokens_seen/1e6:.1f}",
            })
            if len(batch_buf) >= args.tokenizer_batch:
                flush_batch(batch_buf)
                batch_buf = []

        flush_batch(batch_buf)

        final = packer.flush()
        if final is not None:
            if args.format == "arrow":
                writer.write_block(final)
            else:
                if random.random() < args.val_split:
                    val_writer.write_block(final)
                else:
                    train_writer.write_block(final)
            total_blocks_written += 1

    except KeyboardInterrupt:
        print("\n\n[Interrupted] Flushing current shard...")
    finally:
        pbar.close()

    # ── Finalise ──────────────────────────────────────────────────────────────
    print(f"\nFinalising...")

    if args.format == "arrow":
        total_written = writer.close()
        all_shards    = sorted(Path(stage_dir).glob("all_shard_*.arrow"))

        n_val = (
            args.val_shards if args.val_shards > 0
            else max(1, int(len(all_shards) * args.val_split))
        )

        rng = random.Random(args.seed)
        shard_list = list(all_shards)
        rng.shuffle(shard_list)
        val_src   = shard_list[:n_val]
        train_src = shard_list[n_val:]

        train_shards, val_shards = [], []
        for i, src in enumerate(train_src):
            dst = os.path.join(args.output_dir, f"train_shard_{i:05d}.arrow")
            os.rename(str(src), dst)
            train_shards.append(dst)
        for i, src in enumerate(val_src):
            dst = os.path.join(args.output_dir, f"val_shard_{i:05d}.arrow")
            os.rename(str(src), dst)
            val_shards.append(dst)
        try:
            os.rmdir(stage_dir)
        except OSError:
            pass

        stats = {
            "vocab_size":   tokenizer.vocab_size,
            "total_blocks": total_written,
            "total_tokens": total_written * args.block_size,
            "total_docs":   packer.docs_seen,
        }
        write_manifest(args.output_dir, train_shards, val_shards, args, stats)

    else:
        train_total = train_writer.close()
        val_total   = val_writer.close()
        stats = {
            "vocab_size":   tokenizer.vocab_size,
            "total_blocks": total_blocks_written,
            "total_tokens": train_total + val_total,
            "total_docs":   packer.docs_seen,
        }
        write_manifest(
            args.output_dir,
            [os.path.join(args.output_dir, "train.bin")],
            [os.path.join(args.output_dir, "val.bin")],
            args, stats,
        )

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  Tokenization complete")
    print(f"{'='*65}")
    print(f"  Documents processed : {packer.docs_seen:,}")
    print(f"  Tokens seen         : {packer.tokens_seen:,}"
          f"  ({packer.tokens_seen/1e9:.3f}B)")
    print(f"  Blocks written      : {total_blocks_written:,}")
    print(f"  Output              : {args.output_dir}")


if __name__ == "__main__":
    main()
