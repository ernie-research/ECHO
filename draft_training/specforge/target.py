# coding=utf-8
"""Helpers for loading the locally validated the target-5 HF checkpoint."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Optional

DEFAULT_TARGET_ROOT = (
    "<WORKSPACE>/"
    "paddle_hf_alignment"
)


def acceptance_root() -> str:
    return os.environ.get("TARGET_ACCEPTANCE_ROOT", DEFAULT_TARGET_ROOT)


def _ensure_acceptance_importable() -> str:
    src_dir = str(Path(acceptance_root()) / "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    return src_dir


def register_target() -> bool:
    """Register the local the target config with Transformers when available."""
    try:
        _ensure_acceptance_importable()
        from modeling import TargetConfig
        from transformers import AutoConfig

        try:
            AutoConfig.register("echo_target", TargetConfig)
        except ValueError as exc:
            if "already used" not in str(exc):
                raise
        return True
    except (ImportError, FileNotFoundError):
        return False


def is_target_checkpoint(model_path: str) -> bool:
    config_path = Path(model_path) / "config.json"
    if not config_path.is_file():
        return False
    try:
        with config_path.open() as f:
            return json.load(f).get("model_type") == "echo_target"
    except (OSError, ValueError):
        return False


def tokenizer_path(model_path: str, explicit_path: Optional[str] = None) -> str:
    if explicit_path:
        return explicit_path
    if is_target_checkpoint(model_path):
        return str(Path(acceptance_root()) / "tokenizer_hf")
    return model_path


def resolve_mask_token_id(
    tokenizer,
    model_path: str,
    explicit_id: Optional[int] = None,
    embedding_vocab_size: Optional[int] = None,
) -> int:
    """Resolve a valid in-vocabulary noise token without growing the tokenizer.

    the target's validated tokenizer names id 2 (``<|TARGET_MASK|>``) as its EOS
    token instead of exposing a separate ``mask_token``.  Adding a new token
    here would create id ``vocab_size`` while the frozen target embedding still
    has the original vocabulary shape.
    """
    if embedding_vocab_size is None:
        embedding_vocab_size = getattr(tokenizer, "vocab_size", None)
    if embedding_vocab_size is None:
        raise ValueError("embedding_vocab_size is required to validate mask_token_id")

    def validate(token_id: Optional[int], source: str) -> int:
        if token_id is None or not 0 <= int(token_id) < int(embedding_vocab_size):
            raise ValueError(
                f"{source}={token_id} is outside embedding vocabulary "
                f"[0, {embedding_vocab_size})"
            )
        return int(token_id)

    if explicit_id is not None:
        return validate(explicit_id, "mask_token_id")

    tokenizer_mask_id = getattr(tokenizer, "mask_token_id", None)
    if tokenizer_mask_id is not None:
        return validate(tokenizer_mask_id, "tokenizer.mask_token_id")

    # The acceptance tokenizer deliberately uses the model's id-2
    # <|TARGET_MASK|> token as the generation terminator and draft filler.
    if is_target_checkpoint(model_path):
        eos_id = getattr(tokenizer, "eos_token_id", None)
        if eos_id is not None:
            return validate(eos_id, "the target tokenizer.eos_token_id")

    # Generic models may have an unused slot in the target embedding.
    tokenizer_len = len(tokenizer)
    if tokenizer_len < embedding_vocab_size:
        return validate(tokenizer_len, "unused tokenizer slot")

    for name in ("pad_token_id", "eos_token_id", "unk_token_id"):
        token_id = getattr(tokenizer, name, None)
        if token_id is not None:
            return validate(token_id, f"tokenizer.{name}")

    raise ValueError(
        "Could not resolve an in-vocabulary mask token. Pass --mask-token-id "
        "or use a tokenizer with mask/pad/eos/unk token."
    )


def build_target_draft_config(
    target_config,
    *,
    num_hidden_layers: int,
    block_size: int,
    attention_backend: str,
    dspark_config: Optional[dict] = None,
):
    """Build target-layer-typed DSpark draft config around a target.

    DeepSpec implements DSpark per target family (modeling/dspark/qwen3/,
    gemma4/, ...): the draft backbone reuses the target family's decoder
    layer type and only the heads/loss are shared.  This returns the target
    member of that family (``TargetDSparkDraftModel``); dimensions are
    copied from the target, the backbone itself is built from the
    acceptance package's the target modules.
    """
    from specforge.modeling.draft.target_dspark import TargetDSparkDraftConfig

    draft_config = TargetDSparkDraftConfig(
        vocab_size=int(target_config.vocab_size),
        hidden_size=int(target_config.hidden_size),
        intermediate_size=int(
            getattr(target_config, "intermediate_size", 4 * target_config.hidden_size)
        ),
        num_hidden_layers=int(num_hidden_layers),
        num_attention_heads=int(target_config.num_attention_heads),
        num_key_value_heads=int(
            getattr(target_config, "num_key_value_heads", target_config.num_attention_heads)
        ),
        head_dim=int(
            getattr(
                target_config,
                "head_dim",
                target_config.hidden_size // target_config.num_attention_heads,
            )
        ),
        max_position_embeddings=int(target_config.max_position_embeddings),
        q_lora_rank=int(getattr(target_config, "q_lora_rank", 512)),
        kv_lora_rank=int(getattr(target_config, "kv_lora_rank", 512)),
        qk_rope_head_dim=int(getattr(target_config, "qk_rope_head_dim", 64)),
        qk_nope_head_dim=int(getattr(target_config, "qk_nope_head_dim", 192)),
        v_head_dim=int(getattr(target_config, "v_head_dim", 256)),
        moe_intermediate_size=int(getattr(target_config, "moe_intermediate_size", 1024)),
        moe_latent_size=int(getattr(target_config, "moe_latent_size", 512)),
        n_routed_experts=int(getattr(target_config, "n_routed_experts", 256)),
        num_experts_per_tok=int(getattr(target_config, "num_experts_per_tok", 8)),
        n_shared_experts=int(getattr(target_config, "n_shared_experts", 1)),
        rms_norm_eps=float(target_config.rms_norm_eps),
        rope_theta=float(getattr(target_config, "rope_theta", 10000.0)),
        block_size=int(block_size),
        num_target_layers=int(target_config.num_hidden_layers),
        dspark_config=dict(dspark_config or {}),
    )
    draft_config.architectures = ["TargetDSparkDraftModel"]
    draft_config._attn_implementation = attention_backend
    return draft_config


def build_target_dflash_draft_config(
    target_config,
    *,
    num_hidden_layers: int,
    block_size: int,
    attention_backend: str,
    dflash_config: Optional[dict] = None,
):
    """the target member of the DFlash family.

    Same geometry and the same decoder backbone as the DSpark draft (the layer
    stack is literally ``TargetDSparkLayer``), so this reuses
    ``build_target_draft_config`` for the dimension copying and only swaps the
    config class / architectures and drops the Markov+confidence head settings.
    One source for the geometry is what keeps the dflash and dspark arms
    comparable.
    """
    from specforge.modeling.draft.target_dflash import TargetDFlashDraftConfig

    base = build_target_draft_config(
        target_config,
        num_hidden_layers=num_hidden_layers,
        block_size=block_size,
        attention_backend=attention_backend,
        dspark_config=None,
    )
    raw = base.to_dict()
    raw.pop("dspark_config", None)
    raw.pop("architectures", None)
    raw["dflash_config"] = dict(dflash_config or {})
    draft_config = TargetDFlashDraftConfig.from_dict(raw)
    draft_config.architectures = ["TargetDFlashDraftModel"]
    draft_config._attn_implementation = attention_backend
    return draft_config


def load_target(
    model_path: str,
    device: str,
    dtype,
    *,
    use_mtp: bool = False,
):
    """Load the validated target, optionally preserving Echo shared MTP."""
    _ensure_acceptance_importable()
    from modeling.torch_weight_loader import build_and_load_torch

    model, config, stats = build_and_load_torch(
        model_path,
        device=device or "cuda",
        dtype=dtype,
        no_mtp=not use_mtp,
        preserve_mtp_sharing=use_mtp,
    )
    model.model.prepare_inference_weights()
    model.eval()
    model.requires_grad_(False)
    return model, config, stats
