"""
configuration_vortex.py
=======================
HuggingFace-compatible configuration for the Vortex hybrid architecture.

Vortex combines:
  - Parallax dual-track design (two independent processing streams with cross-track swap)
  - Mamba 2 SSD (State Space Duality) blocks replacing attention + FFN

Author: Bryan K Reinhart
License: AGPL-3.0
"""

from transformers import PretrainedConfig

class VortexConfig(PretrainedConfig):
    r"""
    Configuration class for the Vortex hybrid SSM language model.
    
    Vortex is a dual-track State Space Model (SSM) that fuses the Parallax
    cross-track swap mechanism with Mamba 2 SSD blocks. Each track processes
    the input independently through a stack of SSD layers, then the tracks
    exchange their hidden states (the "swap") before a second pass. This gives
    the model two offset vantage points — like the optical concept of parallax —
    while leveraging the linear-complexity, hardware-efficient SSD operator from
    Mamba 2 for sequence modeling.

    Architecture summary
    --------------------
    Pass 1:  x --> Track A (n_layer SSD blocks) --> out_a
             x --> Track B (n_layer SSD blocks) --> out_b

    Swap:    in_a' = out_b + x   (B's view fed to A, residual from embedding)
             in_b' = out_a + x   (A's view fed to B, residual from embedding)

    Pass 2:  in_a' --> Track A (shared weights) --> out_a'
             in_b' --> Track B (shared weights) --> out_b'

    Fusion:  (out_a' + out_b') --> RMSNorm --> LM head --> logits
    
    Args
    ----
    vocab_size (int):
        Vocabulary size. Defaults to 32000 (LlamaTokenizer / GPT2 BPE with
        expansion works equally well).
    d_model (int):
        Model (embedding) dimension. Controls the width of both tracks.
        Default 512 gives ~42M parameters at the default depth.
    n_layer (int):
        Number of SSD blocks **per track**. With weight-tied passes the
        effective compute depth is ``n_layer * num_loops``.
    num_loops (int):
        Number of forward passes through each track. At ``num_loops=2`` one
        swap occurs. At ``num_loops=3`` two swaps occur. ``num_loops=1``
        disables the swap entirely (parallel ensemble mode).
    use_swap (bool):
        Master switch for the cross-track swap. If False the model behaves as
        two independent tracks whose outputs are summed (ablation baseline).
    block_size (int):
        Maximum sequence length (context window in tokens).
    d_state (int):
        SSM state dimension (N in Mamba notation). Controls the memory of
        each SSM head. Default 64.
    d_conv (int):
        Local convolution width inside each SSD block. Default 4.
    expand (int):
        Inner-dimension expansion factor: ``d_inner = expand * d_model``.
        Default 2.
    n_heads (int):
        Number of SSM heads per SSD block. Must evenly divide
        ``d_inner = expand * d_model``.
    dt_rank (str | int):
        Rank of the delta (Δ) projection. ``"auto"`` sets it to
        ``ceil(d_model / 16)``. Can be set to an explicit integer.
    dt_min (float):
        Minimum value for the softplus-clamped Δ. Default 0.001.
    dt_max (float):
        Maximum value for the softplus-clamped Δ. Default 0.1.
    dt_init_floor (float):
        Floor for Δ initialisation. Default 1e-4.
    rms_norm_eps (float):
        Epsilon for RMSNorm layers. Default 1e-5.
    dropout (float):
        Dropout probability applied after the output projection of each SSD
        block. 0.0 is recommended for initial training runs.
    bias (bool):
        Whether to include bias terms in linear projections. Default False
        (following modern LLM practice).
    use_triton (bool):
        If True, attempt to use a Triton-accelerated SSD kernel at runtime.
        Falls back automatically to the pure-PyTorch implementation if Triton
        is unavailable or the kernel raises an error.
    pad_vocab_size_multiple (int):
        Round vocab_size up to the nearest multiple of this value for
        efficient matrix-multiply tiling. Default 8.
    tie_embeddings (bool):
        Whether to tie the input embedding weights with the LM head weights.
        Default True (saves ~vocab_size * d_model parameters).
    initializer_range (float):
        Standard deviation for weight initialisation. Default 0.02.
    """
    
    model_type = "vortex"
    
    def __init__(
        self,
        vocab_size: int              = 32000,
        d_model: int                 = 512,
        n_layer: int                 = 8,
        num_loops: int               = 2,
        use_swap: bool               = True,
        block_size: int              = 1024,
        d_state: int                 = 64,
        d_conv: int                  = 4,
        expand: int                  = 2,
        n_heads: int                 = 8,
        dt_rank: str                 = "auto",
        dt_min: float                = 0.001,
        dt_max: float                = 0.1,
        dt_init_floor: float         = 1e-4,
        rms_norm_eps: float          = 1e-5,
        dropout: float               = 0.0,
        bias: bool                   = False,
        use_triton: bool             = True,
        pad_vocab_size_multiple: int = 8,
        tie_embeddings: bool         = True,
        initializer_range: float     = 0.02,
        **kwargs,
    ):
        # ── Vocab padding ─────────────────────────────────────────────────────
        if vocab_size % pad_vocab_size_multiple != 0:
            vocab_size = ((vocab_size // pad_vocab_size_multiple) + 1) * pad_vocab_size_multiple

        self.vocab_size              = vocab_size
        self.d_model                 = d_model
        self.n_layer                 = n_layer
        self.num_loops               = num_loops
        self.use_swap                = use_swap
        self.block_size              = block_size

        # ── SSD / Mamba 2 hyperparameters ─────────────────────────────────────
        self.d_state                 = d_state
        self.d_conv                  = d_conv
        self.expand                  = expand
        self.n_heads                 = n_heads
        self.d_inner                 = expand * d_model

        if dt_rank == "auto":
            import math
            self.dt_rank             = math.ceil(d_model / 16)
        else:
            self.dt_rank             = int(dt_rank)

        self.dt_min                  = dt_min
        self.dt_max                  = dt_max
        self.dt_init_floor           = dt_init_floor

        # ── General ───────────────────────────────────────────────────────────
        self.rms_norm_eps            = rms_norm_eps
        self.dropout                 = dropout
        self.bias                    = bias
        self.use_triton              = use_triton
        self.pad_vocab_size_multiple = pad_vocab_size_multiple
        self.tie_embeddings          = tie_embeddings
        self.initializer_range       = initializer_range

        super().__init__(**kwargs)


    # ── Convenience helpers ────────────────────────────────────────────────────

    def count_parameters(self) -> dict:
        """Return a rough parameter count breakdown (no actual model needed)."""

        emb = self.vocab_size * self.d_model
        d_inner = self.d_inner

        # Per SSD block (two tracks, shared-weight passes do NOT double params)
        # in_proj: d_model -> 2*d_inner + 2*d_state + n_heads
        in_proj = self.d_model * (2 * d_inner + 2 * self.d_state + self.n_heads)
        conv = d_inner * self.d_conv
        x_proj = d_inner * (self.dt_rank + 2 * self.d_state)
        dt_proj = self.dt_rank * d_inner
        out_proj = d_inner * self.d_model
        norms = 2 * self.d_model  # pre-norm + post-norm per block
        per_block = in_proj + conv + x_proj + dt_proj + out_proj + norms

        # Two tracks, n_layer blocks each (passes are weight-tied → no extra)
        track_params = 2 * self.n_layer * per_block

        lm_head = 0 if self.tie_embeddings else self.vocab_size * self.d_model
        final_norm = self.d_model

        total = emb + track_params + lm_head + final_norm

        return {
            "embedding": emb,
            "track_params (2 tracks, weight-tied passes)": track_params,
            "lm_head": lm_head,
            "final_norm": final_norm,
            "total_approx": total,
            "total_M": round(total / 1e6, 2),
        }

    def __repr__(self) -> str:
        counts = self.count_parameters()
        return (
            f"VortexConfig(\n"
            f"  vocab_size={self.vocab_size}, d_model={self.d_model},\n"
            f"  n_layer={self.n_layer}, num_loops={self.num_loops},\n"
            f"  use_swap={self.use_swap}, block_size={self.block_size},\n"
            f"  d_state={self.d_state}, d_inner={self.d_inner},\n"
            f"  n_heads={self.n_heads}, dt_rank={self.dt_rank},\n"
            f"  use_triton={self.use_triton},\n"
            f"  ~params: {counts['total_M']}M\n"
            f")"
        )
