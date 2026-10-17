"""the target V2 PyTorch HF package — auto-discovery entry point.

AutoConfig and AutoModel will pick this up when loading from the checkpoint
directory with trust_remote_code=True.

Usage (from a directory containing modeling/ as a sub-package):
    import sys
    sys.path.insert(0, "<WORKSPACE>/paddle_hf_alignment/src>")
    from modeling import TargetConfig, TargetForCausalLM
"""
from .configuration import TargetConfig
from .modeling import (
    TargetForCausalLM,
    TargetModel,
    TargetMTPHead,
    TargetDecoderLayer,
    TargetAttention,
    TargetMoE,
)

__all__ = [
    "TargetConfig",
    "TargetForCausalLM",
    "TargetModel",
    "TargetMTPHead",
    "TargetDecoderLayer",
    "TargetAttention",
    "TargetMoE",
]
