"""
Vortex model package — registers with the HuggingFace AutoModel ecosystem.
"""

from .configuration_vortex import VortexConfig
from .modeling_vortex import VortexForCausalLM, VortexModel

try:
    from transformers import AutoConfig, AutoModelForCausalLM

    AutoConfig.register("vortex", VortexConfig)
    AutoModelForCausalLM.register(VortexConfig, VortexForCausalLM)
except Exception:
    pass  # registration is optional; direct imports always work

__all__ = ["VortexConfig", "VortexModel", "VortexForCausalLM"]
