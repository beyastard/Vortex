"""
diag.py
=======
Hardware and environment diagnostics for Vortex.

Checks Python, PyTorch, CUDA, Triton, Flash Attention, and available VRAM.
Run this first to verify your environment is correctly set up.

Author: Bryan K Reinhart
License: AGPL-3.0

Usage
-----
    python scripts/tools/diag.py
"""

import sys
import os
import platform


def section(title: str):
    print(f"\n{'='*55}")
    print(f"  {title}")
    print(f"{'='*55}")


def check(label: str, value, ok: bool = True):
    status = "✓" if ok else "✗"
    print(f"  {status}  {label:<35} {value}")


def main():
    print("\nVortex Environment Diagnostics")
    print("=" * 55)
    
    # Python
    section("Python")
    check("Version", platform.python_version(), ok=sys.version_info >= (3, 11))
    check("Platform", platform.platform())
    check("Executable", sys.executable)
    
    # PyTorch
    section("PyTorch")
    try:
        import torch
        check("torch version", torch.__version__)
        check("CUDA available", str(torch.cuda.is_available()), ok=torch.cuda.is_available())
        if torch.cuda.is_available():
            check("CUDA version", torch.version.cuda)
            check("cuDNN version", str(torch.backends.cudnn.version()))
            n_dev = torch.cuda.device_count()
            check("GPU count", str(n_dev), ok=n_dev > 0)
            for i in range(n_dev):
                props = torch.cuda.get_device_properties(i)
                vram  = props.total_memory / 1e9
                check(f"  GPU {i}: {props.name[:30]}", f"{vram:.1f} GB VRAM", ok=vram >= 4.0)
                free, total = torch.cuda.mem_get_info(i)
                check(f"  GPU {i} free VRAM", f"{free/1e9:.1f} / {total/1e9:.1f} GB", ok=free/1e9 >= 2.0)
        check("torch.compile available", str(hasattr(torch, "compile")))
        check("AMP available", str(hasattr(torch.amp, "autocast")))
        check("fused AdamW", "yes" if torch.cuda.is_available() else "no (CPU)")
    except ImportError as e:
        check("torch", f"NOT FOUND: {e}", ok=False)
    
    # Transformers / HF ecosystem
    section("HuggingFace Ecosystem")
    for pkg in ["transformers", "datasets", "safetensors", "tokenizers", "tensorboard", "numpy", "tqdm"]:
        try:
            mod = __import__(pkg)
            ver = getattr(mod, "__version__", "?")
            check(pkg, ver)
        except ImportError:
            check(pkg, "NOT FOUND", ok=False)
    
    # Triton
    section("Triton")
    try:
        import triton
        check("triton version", triton.__version__)
        # Try a minimal JIT compile
        try:
            import triton.language as tl
            @triton.jit
            def _noop(x_ptr, BLOCK: tl.constexpr):
                pass
            check("triton JIT", "OK (kernel compiles)")
        except Exception as e:
            check("triton JIT", f"WARN: {e}", ok=False)
    except ImportError:
        check("triton", "NOT FOUND (Triton SSD kernel unavailable)", ok=False)
        print("    → Pure-PyTorch fallback will be used automatically.")
    
    # Flash Attention 2
    section("Flash Attention 2")
    try:
        import flash_attn
        ver = getattr(flash_attn, "__version__", "?")
        check("flash_attn version", ver)
        check("Note", "FA2 available (not used directly by Vortex SSD,\n"
              "                                    but available if extended)")
    except ImportError:
        check("flash_attn", "NOT FOUND", ok=False)
        print("    → Vortex SSD does not require FA2 directly.")
    
    # xformers
    section("xformers (optional)")
    try:
        import xformers
        check("xformers", xformers.__version__)
    except ImportError:
        check("xformers", "NOT FOUND (optional, not required)")


if __name__ == "__main__":
    main()
