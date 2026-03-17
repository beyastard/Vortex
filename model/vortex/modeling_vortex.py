"""
modeling_vortex.py
==================
Full implementation of the Vortex hybrid language model.

Architecture
------------
Vortex fuses two ideas:

1. **Parallax dual-track swap** — two independent tracks process the same
   input, then exchange hidden states between passes. Each track sees the
   other's "view" of the sequence before refining its own representation.

2. **Mamba 2 SSD blocks** — each track is a stack of State Space Duality
   (SSD) layers. SSD unifies linear attention and SSMs into a single
   hardware-efficient operator (O(L) time and memory vs O(L²) for attention).

Together the model has:
  - Linear-complexity sequence modeling (from SSD)
  - Dual-perspective depth via cross-track swap (from Parallax)
  - Weight-tied multi-pass computation (recurrent character)

Implementation notes
--------------------
The SSD block is implemented in two variants:
  - ``SSDBlockPure``:   pure PyTorch, always available, used as fallback.
  - ``SSDBlockTriton``: Triton-accelerated selective scan kernel.
                        Loaded dynamically; falls back to pure if unavailable.

``VortexSSDBlock`` wraps both and selects at runtime based on
``config.use_triton`` and kernel availability.

Author: Bryan K Reinhart
License: AGPL-3.0
"""

from __future__ import annotations

import math
import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from .configuration_vortex import VortexConfig

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Triton kernel availability check
# ──────────────────────────────────────────────────────────────────────────────

_TRITON_AVAILABLE = False
_triton_selective_scan = None  # will be populated below if available

try:
    import triton  # noqa: F401
    import triton.language as tl  # noqa: F401
    _TRITON_AVAILABLE = True
    logger.info("Triton is available — Triton SSD kernel will be used if requested.")
except ImportError:
    logger.info("Triton not found — falling back to pure-PyTorch SSD implementation.")


# ──────────────────────────────────────────────────────────────────────────────
# Utility: RMSNorm
# ──────────────────────────────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (no bias term)."""

    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).to(x.dtype) * self.weight


# ──────────────────────────────────────────────────────────────────────────────
# Utility: 1-D causal depthwise convolution
# ──────────────────────────────────────────────────────────────────────────────

class CausalDepthwiseConv1d(nn.Module):
    """
    Causal depthwise conv over the sequence dimension.
    Used inside the SSD block to provide local mixing before the SSM.
    """

    def __init__(self, channels: int, kernel_size: int, bias: bool = True):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            groups=channels,
            padding=kernel_size - 1,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, C) -> conv expects (B, C, L)
        x = x.transpose(1, 2)
        x = self.conv(x)
        # Remove future tokens (causal mask via padding trim)
        x = x[:, :, : -self.kernel_size + 1] if self.kernel_size > 1 else x
        return x.transpose(1, 2)


# ──────────────────────────────────────────────────────────────────────────────
# Pure-PyTorch selective scan (SSD core)
# ──────────────────────────────────────────────────────────────────────────────

def selective_scan_pure2(
    u, delta, A, B, C, D,
    delta_bias=None, delta_softplus=True,
):
    """
    Parallel associative scan — O(log L) depth instead of O(L) sequential.
    Runs entirely in vectorised PyTorch ops with no Python loop over tokens.
    """
    dtype_in = u.dtype
    u     = u.float()
    delta = delta.float()

    if delta_bias is not None:
        delta = delta + delta_bias.unsqueeze(0).unsqueeze(0)
    if delta_softplus:
        delta = F.softplus(delta)

    B_seq = B.float()   # (batch, L, d_state)
    C_seq = C.float()   # (batch, L, d_state)
    batch, L, d_inner = u.shape
    d_state = A.shape[1]

    # Discretise
    # deltaA: (batch, L, d_inner, d_state)
    # deltaB_u: (batch, L, d_inner, d_state)
    deltaA   = torch.exp(
        delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0)
    )
    deltaB_u = (
        delta.unsqueeze(-1)
        * B_seq.unsqueeze(2)
        * u.unsqueeze(-1)
    )

    # Parallel associative scan
    # We represent the state at each position as (a_i, b_i) where:
    #   h_i = a_i * h_{i-1} + b_i
    # The associative combination is:
    #   (a_j, b_j) ∘ (a_i, b_i) = (a_j * a_i, a_j * b_i + b_j)
    # This lets us compute all h_i in parallel in O(log L) steps.

    log2_L = math.ceil(math.log2(max(L, 1)))

    # Pad L to next power of 2 for the binary tree reduction
    L_pad = 2 ** log2_L
    pad   = L_pad - L

    # a: (batch, L_pad, d_inner, d_state)
    # b: (batch, L_pad, d_inner, d_state)
    if pad > 0:
        a = F.pad(deltaA,   (0, 0, 0, 0, 0, pad))
        b = F.pad(deltaB_u, (0, 0, 0, 0, 0, pad))
    else:
        a = deltaA
        b = deltaB_u

    # Up-sweep (reduce)
    for d in range(log2_L):
        stride = 2 ** (d + 1)
        left   = torch.arange(0,      L_pad, stride, device=u.device)
        right  = torch.arange(stride//2, L_pad, stride, device=u.device)
        if len(right) == 0:
            break
        a_right = a[:, right]
        b_right = b[:, right]
        a_left  = a[:, left[:len(right)]]
        b_left  = b[:, left[:len(right)]]
        a[:, right] = a_right * a_left
        b[:, right] = a_right * b_left + b_right

    # Down-sweep (scan)
    a[:, -1] = 0.0
    b[:, -1] = 0.0
    for d in range(log2_L - 1, -1, -1):
        stride = 2 ** (d + 1)
        left   = torch.arange(0,        L_pad, stride, device=u.device)
        right  = torch.arange(stride//2, L_pad, stride, device=u.device)
        if len(right) == 0:
            break
        t_a             = a[:, left[:len(right)]].clone()
        t_b             = b[:, left[:len(right)]].clone()
        a[:, left[:len(right)]] = a[:, right]
        b[:, left[:len(right)]] = b[:, right]
        a[:, right] = a[:, right] * t_a
        b[:, right] = a[:, right] * t_b + b[:, right]

    # h contains the prefix-scan results: h_i = sum_{j<=i} (prod_{k>j}^i a_k) * b_j
    # Trim padding
    h = b[:, :L]   # (batch, L, d_inner, d_state)

    # Output: y_t = C_t · h_t + D · u_t
    # C_seq: (batch, L, d_state) -> (batch, L, 1, d_state)
    y = (h * C_seq.unsqueeze(2)).sum(dim=-1)   # (batch, L, d_inner)
    y = y + u * D.unsqueeze(0).unsqueeze(0)

    return y.to(dtype_in)

def selective_scan_pure(
    u: torch.Tensor,      # (B, L, d_inner)
    delta: torch.Tensor,  # (B, L, d_inner)
    A: torch.Tensor,      # (d_inner, d_state) log-parameterised
    B: torch.Tensor,      # (B, L, d_state)
    C: torch.Tensor,      # (B, L, d_state)
    D: torch.Tensor,      # (d_inner,) skip connection
    delta_bias: Optional[torch.Tensor] = None,
    delta_softplus: bool = True,
) -> torch.Tensor:
    """
    Pure-PyTorch selective scan — O(L·d_inner·d_state) time.

    Implements the discretised SSM recurrence:
        h_t = A_bar_t · h_{t-1} + B_bar_t · u_t
        y_t = C_t · h_t + D · u_t

    where A_bar and B_bar are ZOH (zero-order hold) discretisations of A and B
    using the step size delta_t.
    """
    dtype_in = u.dtype
    u = u.float()
    delta = delta.float()
    if delta_bias is not None:
        delta = delta + delta_bias.unsqueeze(0).unsqueeze(0)
    if delta_softplus:
        delta = F.softplus(delta)

    B_seq = B.float()
    C_seq = C.float()

    batch, seqlen, d_inner = u.shape
    d_state = A.shape[1]

    # Discretise: A_bar = exp(delta * A),  B_bar = delta * B (ZOH approx for B)
    # A: (d_inner, d_state), delta: (batch, L, d_inner)
    deltaA = torch.exp(delta.unsqueeze(-1) * A)           # (B, L, d_inner, d_state)
    deltaB_u = delta.unsqueeze(-1) * B_seq.unsqueeze(2) * u.unsqueeze(-1)
    # deltaB_u: (B, L, d_inner, d_state)

    # Sequential scan
    h = torch.zeros(batch, d_inner, d_state, device=u.device, dtype=torch.float32)
    ys = []
    for t in range(seqlen):
        h = deltaA[:, t] * h + deltaB_u[:, t]             # (B, d_inner, d_state)
        y = (h * C_seq[:, t].unsqueeze(1)).sum(dim=-1)    # (B, d_inner)
        ys.append(y)

    y = torch.stack(ys, dim=1)                            # (B, L, d_inner)
    y = y + u * D
    return y.to(dtype_in)


# ──────────────────────────────────────────────────────────────────────────────
# Triton selective scan (loaded lazily)
# ──────────────────────────────────────────────────────────────────────────────

def _build_triton_selective_scan2():
    """
    Triton-accelerated parallel prefix scan using the associative scan algorithm.
    Each Triton kernel handles one (batch, d_inner, d_state) lane across the
    sequence in parallel — no Python loop over tokens.
    """
    if not _TRITON_AVAILABLE:
        return None

    try:
        import triton
        import triton.language as tl

        @triton.jit
        def _parallel_scan_kernel(
            a_ptr, b_ptr, out_ptr,
            L, d_inner, d_state,
            stride_bl, stride_bd, stride_bs,
            BLOCK_L: tl.constexpr,
        ):
            """
            One program instance per (batch * d_inner * d_state) lane.
            Processes BLOCK_L tokens per instance using a work-efficient
            parallel scan within the block.
            """
            pid    = tl.program_id(0)
            n_lane = d_inner * d_state
            bid    = pid // n_lane
            lid    = pid  % n_lane
            did    = lid  // d_state
            sid    = lid  % d_state

            offsets = tl.arange(0, BLOCK_L)
            mask    = offsets < L

            base = bid * stride_bl * L + did * stride_bd + sid * stride_bs

            a_vals = tl.load(a_ptr + base + offsets * stride_bl,
                             mask=mask, other=1.0).to(tl.float32)
            b_vals = tl.load(b_ptr + base + offsets * stride_bl,
                             mask=mask, other=0.0).to(tl.float32)

            # Sequential scan within the block (fast in SRAM)
            h = 0.0
            for i in tl.static_range(BLOCK_L):
                active = i < L
                h = tl.where(active, a_vals[i] * h + b_vals[i], h)
                tl.store(out_ptr + base + i * stride_bl, h,
                         mask=(i < L))

        def selective_scan_triton(u, delta, A, B, C, D,
                                   delta_bias=None, delta_softplus=True):
            dtype_in = u.dtype
            u     = u.float()
            delta = delta.float()

            if delta_bias is not None:
                delta = delta + delta_bias.unsqueeze(0).unsqueeze(0)
            if delta_softplus:
                delta = F.softplus(delta)

            batch, L, d_inner = u.shape
            d_state = A.shape[1]

            deltaA   = torch.exp(
                delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0)
            )
            deltaB_u = (
                delta.unsqueeze(-1)
                * B.float().unsqueeze(2)
                * u.unsqueeze(-1)
            )

            # Layout: (batch, L, d_inner, d_state) — contiguous
            a = deltaA.contiguous()
            b = deltaB_u.contiguous()
            h = torch.zeros_like(b)

            stride_bl = a.stride(1)
            stride_bd = a.stride(2)
            stride_bs = a.stride(3)

            BLOCK_L   = triton.next_power_of_2(L)
            n_programs = batch * d_inner * d_state

            _parallel_scan_kernel[(n_programs,)](
                a, b, h,
                L, d_inner, d_state,
                stride_bl, stride_bd, stride_bs,
                BLOCK_L=BLOCK_L,
            )

            y = (h * C.float().unsqueeze(2)).sum(dim=-1)
            y = y + u * D.unsqueeze(0).unsqueeze(0)
            return y.to(dtype_in)

        return selective_scan_triton

    except Exception as e:
        logger.warning(f"Triton parallel scan build failed ({e}), using pure-PyTorch.")
        return None

def _build_triton_selective_scan():
    """
    Define and compile the Triton-accelerated selective scan kernel.
    Returns a callable with the same signature as ``selective_scan_pure``,
    or None if compilation fails.
    """
    if not _TRITON_AVAILABLE:
        return None

    try:
        import triton
        import triton.language as tl

        @triton.jit
        def _selective_scan_kernel(
            u_ptr, delta_ptr, A_ptr, B_ptr, C_ptr, D_ptr,
            out_ptr,
            B_size, L_size, d_inner_size, d_state_size,
            BLOCK_D: tl.constexpr,
        ):
            """
            Each program instance handles one (batch, d_inner) lane.
            Iterates over sequence positions performing the SSM recurrence.
            """
            pid_b = tl.program_id(0)
            pid_d = tl.program_id(1)

            d_offsets = tl.arange(0, BLOCK_D)
            mask_d = d_offsets < d_state_size

            # h accumulator: one value per state dim
            h = tl.zeros([BLOCK_D], dtype=tl.float32)

            for t in range(L_size):
                # Load u[b, t, pid_d] and delta[b, t, pid_d]
                u_idx = pid_b * L_size * d_inner_size + t * d_inner_size + pid_d
                d_idx = u_idx
                u_val = tl.load(u_ptr + u_idx).to(tl.float32)
                delta_val = tl.load(delta_ptr + d_idx).to(tl.float32)
                delta_val = tl.math.log(1.0 + tl.math.exp(delta_val))  # softplus

                # Load A[pid_d, :d_state], B[b, t, :d_state], C[b, t, :d_state]
                A_base = pid_d * d_state_size + d_offsets
                B_base = pid_b * L_size * d_state_size + t * d_state_size + d_offsets
                C_base = B_base

                A_vals = tl.load(A_ptr + A_base, mask=mask_d, other=0.0).to(tl.float32)
                B_vals = tl.load(B_ptr + B_base, mask=mask_d, other=0.0).to(tl.float32)
                C_vals = tl.load(C_ptr + C_base, mask=mask_d, other=0.0).to(tl.float32)
                D_val  = tl.load(D_ptr + pid_d).to(tl.float32)

                dA = tl.math.exp(delta_val * A_vals)
                dBu = delta_val * B_vals * u_val

                h = dA * h + dBu
                y = tl.sum(h * C_vals) + D_val * u_val

                out_idx = u_idx
                tl.store(out_ptr + out_idx, y.to(tl.float16))

        def selective_scan_triton(u, delta, A, B, C, D, delta_bias=None, delta_softplus=True):
            B_size, L_size, d_inner_size = u.shape
            d_state_size = A.shape[1]

            if delta_bias is not None:
                delta = delta + delta_bias.unsqueeze(0).unsqueeze(0)

            out = torch.empty_like(u)
            BLOCK_D = triton.next_power_of_2(d_state_size)

            grid = (B_size, d_inner_size)
            _selective_scan_kernel[grid](
                u, delta, A, B, C, D, out,
                B_size, L_size, d_inner_size, d_state_size,
                BLOCK_D=BLOCK_D,
            )
            return out

        return selective_scan_triton

    except Exception as e:
        logger.warning(f"Triton kernel compilation failed ({e}), using pure-PyTorch.")
        return None


# ──────────────────────────────────────────────────────────────────────────────
# SSD Block (one Mamba-2-style layer)
# ──────────────────────────────────────────────────────────────────────────────

class VortexSSDBlock(nn.Module):
    """
    A single Mamba-2 SSD (State Space Duality) block.

    Structure (following the Mamba 2 paper):
      x_in  (B, L, d_model)
        |
      RMSNorm
        |
      in_proj  -> z (gate) and x (SSM input) and B, C projections, dt
        |
      CausalConv1d on x
        |
      SiLU activation on x
        |
      Selective Scan (SSD core) with parameters A, B, C, dt
        |
      Gated output: y * silu(z)
        |
      out_proj -> (B, L, d_model)
        |
      Dropout
        |
      + x_in (residual)

    The selective scan is routed to the Triton kernel if available and
    ``use_triton=True`` in config, otherwise falls back to pure PyTorch.
    """

    def __init__(self, config: VortexConfig):
        super().__init__()
        self.config = config
        d_model   = config.d_model
        d_inner   = config.d_inner    # expand * d_model
        d_state   = config.d_state
        d_conv    = config.d_conv
        dt_rank   = config.dt_rank
        bias      = config.bias

        # ── Norms ─────────────────────────────────────────────────────────────
        self.norm = RMSNorm(d_model, eps=config.rms_norm_eps)

        # ── Input projection ──────────────────────────────────────────────────
        # Projects to: x (d_inner), z (d_inner), B (d_state), C (d_state), dt (dt_rank)
        self.in_proj = nn.Linear(
            d_model,
            2 * d_inner + 2 * d_state + dt_rank,
            bias=bias,
        )

        # ── Local conv on x ───────────────────────────────────────────────────
        self.conv1d = CausalDepthwiseConv1d(d_inner, d_conv, bias=True)

        # ── SSM parameters ────────────────────────────────────────────────────
        # A: initialised with log of a Hippo-like matrix (uniform here for simplicity)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).repeat(d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))   # log-parameterised for positivity
        self.D = nn.Parameter(torch.ones(d_inner))

        # Δ (dt) projection and bias
        self.dt_proj = nn.Linear(dt_rank, d_inner, bias=True)

        # Initialise dt_proj bias so exp(bias) is in [dt_min, dt_max]
        dt = torch.exp(
            torch.rand(d_inner) * (math.log(config.dt_max) - math.log(config.dt_min))
            + math.log(config.dt_min)
        ).clamp(min=config.dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

        # ── Output projection ─────────────────────────────────────────────────
        self.out_proj = nn.Linear(d_inner, d_model, bias=bias)
        self.dropout  = nn.Dropout(config.dropout)

        # ── Triton kernel (resolved once at first forward) ────────────────────
        self._triton_fn = None
        self._triton_resolved = False

    def _get_scan_fn(self):
        """Lazily resolve the scan function to use (Triton or pure PyTorch)."""
        if not self._triton_resolved:
            if self.config.use_triton and _TRITON_AVAILABLE:
                fn = _build_triton_selective_scan()
                self._triton_fn = fn  # None if build failed
            self._triton_resolved = True
        return self._triton_fn if self._triton_fn is not None else selective_scan_pure

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            (batch, seq_len, d_model)
        """
        residual = x
        x = self.norm(x)

        B, L, _ = x.shape

        # ── Project ───────────────────────────────────────────────────────────
        proj = self.in_proj(x)
        # Split: xz (2*d_inner), B_seq (d_state), C_seq (d_state), dt (dt_rank)
        d_inner = self.config.d_inner
        d_state = self.config.d_state
        dt_rank = self.config.dt_rank

        xz,   rest = proj.split([2 * d_inner, proj.shape[-1] - 2 * d_inner], dim=-1)
        B_seq, rest = rest.split([d_state, rest.shape[-1] - d_state], dim=-1)
        C_seq, dt   = rest.split([d_state, dt_rank], dim=-1)

        x_part, z = xz.chunk(2, dim=-1)   # each (B, L, d_inner)

        # ── Local conv + activation ───────────────────────────────────────────
        x_part = self.conv1d(x_part)
        x_part = F.silu(x_part)

        # ── Delta ─────────────────────────────────────────────────────────────
        delta = self.dt_proj(dt)   # (B, L, d_inner)

        # ── Selective scan ────────────────────────────────────────────────────
        A = -torch.exp(self.A_log.float())   # (d_inner, d_state), negative
        scan_fn = self._get_scan_fn()
        y = scan_fn(
            x_part, delta, A, B_seq, C_seq, self.D,
            delta_bias=None,
            delta_softplus=True,
        )

        # ── Gated output ──────────────────────────────────────────────────────
        y = y * F.silu(z)

        # ── Output projection + residual ──────────────────────────────────────
        out = self.out_proj(y)
        out = self.dropout(out)
        return out + residual


# ──────────────────────────────────────────────────────────────────────────────
# Single track: a stack of SSD blocks
# ──────────────────────────────────────────────────────────────────────────────

class VortexTrack(nn.Module):
    """
    One of the two parallel tracks in the Vortex architecture.
    Consists of ``n_layer`` SSD blocks with independent weights.
    Weights are *reused* across passes (weight-tied recurrence), which means
    this module's parameters do not multiply with ``num_loops``.
    """

    def __init__(self, config: VortexConfig):
        super().__init__()
        self.blocks = nn.ModuleList([VortexSSDBlock(config) for _ in range(config.n_layer)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


# ──────────────────────────────────────────────────────────────────────────────
# Full Vortex model
# ──────────────────────────────────────────────────────────────────────────────

class VortexModel(PreTrainedModel):
    """
    Vortex base model (no LM head).

    Returns the final hidden states after dual-track processing and fusion.
    """

    config_class = VortexConfig
    base_model_prefix = "vortex"
    supports_gradient_checkpointing = True

    def __init__(self, config: VortexConfig):
        super().__init__(config)
        self.config = config

        # ── Token embedding (shared by both tracks) ───────────────────────────
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)

        # ── Dual tracks ───────────────────────────────────────────────────────
        self.track_a = VortexTrack(config)
        self.track_b = VortexTrack(config)

        # ── Final fusion norm ─────────────────────────────────────────────────
        self.norm_f = RMSNorm(config.d_model, eps=config.rms_norm_eps)

        # ── Dropout on embedding ─────────────────────────────────────────────
        self.drop = nn.Dropout(config.dropout)

        self.post_init()

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)

    def get_input_embeddings(self):
        return self.embedding

    def set_input_embeddings(self, value):
        self.embedding = value

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,  # accepted but not used (SSMs are causal)
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            input_ids: (batch, seq_len)
        Returns:
            hidden: (batch, seq_len, d_model)
        """
        x = self.drop(self.embedding(input_ids))  # (B, L, d_model)

        # ── Multi-loop dual-track processing ──────────────────────────────────
        out_a = x
        out_b = x

        for loop_idx in range(self.config.num_loops):
            # Both tracks process in parallel
            new_a = self.track_a(out_a)
            new_b = self.track_b(out_b)

            # Cross-track swap (if enabled and not the last pass)
            if self.config.use_swap and loop_idx < self.config.num_loops - 1:
                # Each track receives the *other* track's output + original embedding
                out_a = new_b + x   # B's view goes into A
                out_b = new_a + x   # A's view goes into B
            else:
                out_a = new_a
                out_b = new_b

        # ── Fusion ────────────────────────────────────────────────────────────
        hidden = self.norm_f(out_a + out_b)
        return hidden


class VortexForCausalLM(PreTrainedModel):
    """
    Vortex model with a causal language-modelling head.

    This is the main class to use for training and inference.
    It wraps ``VortexModel`` and adds a linear projection to vocabulary logits,
    with optional weight-tying between the embedding and the LM head.
    """

    config_class = VortexConfig
    base_model_prefix = "vortex"
    supports_gradient_checkpointing = True
    # tied weights now handled manually
    #_tied_weights_keys = ["lm_head.weight", "vortex.embedding.weight"]  # ← BOTH keys
    #_tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: VortexConfig):
        super().__init__(config)
        self.vortex = VortexModel(config)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        if config.tie_embeddings:
            self.lm_head.weight = self.vortex.embedding.weight

        self.post_init()

    def _init_weights(self, module: nn.Module):
        self.vortex._init_weights(module)

    def get_input_embeddings(self):
        return self.vortex.embedding

    def set_input_embeddings(self, value):
        self.vortex.embedding = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings
    
    def tie_weights(self, **kwargs):
        if self.config.tie_embeddings:
            self.lm_head.weight = self.vortex.embedding.weight
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        # Use strict=False so missing lm_head.weight (tied, not saved) doesn't error
        kwargs.setdefault("ignore_mismatched_sizes", False)
        model = super().from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
        
        # Re-tie after load
        if model.config.tie_embeddings:
            model.lm_head.weight = model.vortex.embedding.weight
        return model

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        """
        Args:
            input_ids:      (batch, seq_len)
            attention_mask: (batch, seq_len) — accepted for HF compatibility,
                            not used internally (SSD is inherently causal).
            labels:         (batch, seq_len) — if provided, loss is computed.
                            Tokens are shifted internally (standard CLM convention).

        Returns:
            CausalLMOutputWithPast with fields: loss, logits.
        """
        hidden = self.vortex(input_ids, attention_mask=attention_mask)
        logits = self.lm_head(hidden)  # (B, L, vocab_size)

        loss = None
        if labels is not None:
            # Shift: predict token t+1 from token t
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return CausalLMOutputWithPast(loss=loss, logits=logits)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor,
        max_new_tokens: int = 200,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.0,
        eos_token_id: Optional[int] = None,
    ) -> torch.LongTensor:
        """
        Simple autoregressive generation with temperature, top-k, and top-p sampling.

        Args:
            input_ids:          (1, prompt_len) — batch size 1 only.
            max_new_tokens:     maximum tokens to generate.
            temperature:        sampling temperature (1.0 = unmodified).
            top_k:              keep only top-k logits before sampling.
            top_p:              nucleus sampling probability mass threshold.
            repetition_penalty: penalise already-seen tokens (>1.0 penalises).
            eos_token_id:       stop generation when this token is sampled.

        Returns:
            (1, prompt_len + generated_len) token ids.
        """
        self.eval()
        generated = input_ids.clone()
        block_size = self.config.block_size

        for _ in range(max_new_tokens):
            # Trim context to block_size
            ctx = generated[:, -block_size:]

            with torch.amp.autocast(
                device_type=ctx.device.type,
                dtype=torch.float16,
                enabled=(ctx.device.type == "cuda")
            ):
                out = self(ctx)
            logits = out.logits[:, -1, :].float()  # (1, vocab)

            # Repetition penalty
            if repetition_penalty != 1.0:
                for token_id in generated[0].tolist():
                    if logits[0, token_id] > 0:
                        logits[0, token_id] /= repetition_penalty
                    else:
                        logits[0, token_id] *= repetition_penalty

            # Temperature
            logits = logits / max(temperature, 1e-8)

            # Top-k filtering
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, -1:]] = float("-inf")

            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
                # Remove tokens beyond nucleus
                sorted_remove = cumulative_probs - sorted_logits.softmax(dim=-1) > top_p
                sorted_logits[sorted_remove] = float("-inf")
                logits = torch.zeros_like(logits).scatter_(1, sorted_idx, sorted_logits)

            probs = logits.softmax(dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)

            if eos_token_id is not None and next_token.item() == eos_token_id:
                break

        return generated
