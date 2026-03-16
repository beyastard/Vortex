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
    
    Args
    ----
    vocab_size (int):
        Vocabulary size. Defaults to 32000 (LlamaTokenizer / GPT2 BPE with
        expansion works equally well).
    d_model (int):
        Model (embedding) dimension. Controls the width of both tracks.
        Default 512 gives ~42M parameters at the default depth.
    """
    
    model_type = "vortex"
    
    def __init__(
        self,
        vocab_size: int              = 32000,
        d_model: int                 = 512,
        pad_vocab_size_multiple: int = 8,
    ):
        # vocab padding
        if vocab_size % pad_vocab_size_multiple != 0:
            vocab_size = ((vocab_size // pad_vocab_size_multiple) + 1) * pad_vocab_size_multiple
        
        self.vocab_size              = vocab_size
        self.d_model                 = d_model
