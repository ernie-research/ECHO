# coding=utf-8
"""target-layer-typed DFlash draft (same backbone as the target DSpark draft).

``target_dspark`` already ports the target family decoder layer onto the DFlash
dual-source key/value contract (queries from the draft tokens, keys/values from
``cat(fc(target_capture), draft_hidden)`` -- exactly ``Qwen3DFlashAttention``'s
k_ctx/k_noise).  DSpark = that backbone + Markov/confidence heads; plain DFlash
is the same backbone with no extra head (``DFlashDraftModel._init_draft_head``
is a no-op).  So this module reuses ``TargetDSparkLayer`` verbatim and only
swaps the base class, which keeps the two arms' feature routing and backbone
bit-identical and leaves the method as the single difference.

Budget: ``block_size`` positions go through the draft backbone; without
``anchor_prediction`` position 0 predicts the anchor itself and is dropped from
the loss, so ``block_size - 1`` draft tokens are emitted and the target verifies
``block_size`` positions (train_dflash.py:614-629).  block_size=8 therefore
matches the DSpark block-7 arm (7 draft / 8 verified) and eagle3's ttt_length=7.

Every draft parameter is cold-started; no target MTP/decoder weights are loaded.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from specforge.target import register_target

from .base import Eagle3DraftModel
from .dflash import DFlashDraftModel, build_target_layer_ids
from .target_dspark import TargetDSparkDraftConfig, TargetDSparkLayer
from .registry import register_draft

from modeling import TargetConfig
from modeling.modeling import (
    TargetRMSNorm,
    TargetRotaryEmbedding,
)


def _build_target_config(config) -> "TargetConfig":
    """Verbatim copy of the target_config built by TargetDSparkDraftModel, so the
    two arms share one backbone definition."""
    return TargetConfig(
        vocab_size=config.vocab_size,
        hidden_size=config.hidden_size,
        num_hidden_layers=config.num_hidden_layers,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        max_position_embeddings=config.max_position_embeddings,
        q_lora_rank=config.q_lora_rank,
        kv_lora_rank=config.kv_lora_rank,
        qk_rope_head_dim=config.qk_rope_head_dim,
        qk_nope_head_dim=config.qk_nope_head_dim,
        v_head_dim=config.v_head_dim,
        moe_intermediate_size=config.moe_intermediate_size,
        moe_latent_size=config.moe_latent_size,
        n_routed_experts=config.n_routed_experts,
        num_experts_per_tok=config.num_experts_per_tok,
        n_shared_experts=config.n_shared_experts,
        n_group=1,
        topk_group=1,
        topk_method="noaux_tc",
        scoring_func="sigmoid",
        routed_scaling_factor=2.5,
        routed_scaling_factor_learnable=True,
        multi_latent_attention=True,
        use_qk_norm=True,
        qk_norm_type="per_head",
        use_gated_attn=True,
        use_attn_merger=True,
        rope_type="rope",
        rotary_dim=config.qk_rope_head_dim,
        rope_interleave=config.rope_interleave,
        first_k_dense_replace=0,
        moe_layer_freq=1,
        rms_norm_eps=config.rms_norm_eps,
        rope_theta=config.rope_theta,
        use_mtp=False,
    )


class TargetDFlashDraftConfig(TargetDSparkDraftConfig):
    """Same geometry as the DSpark draft config; block_size default is the
    budget-matched 8 (7 draft tokens + the anchor position)."""

    model_type = "target_dflash_draft"

    def __init__(self, block_size: int = 8, **kwargs):
        super().__init__(block_size=block_size, **kwargs)


@register_draft
class TargetDFlashDraftModel(DFlashDraftModel):
    """DFlash draft with a target-5 decoder backbone and no extra head."""

    config_class = TargetDFlashDraftConfig
    _no_split_modules = ["TargetDSparkLayer"]
    architectures = ["TargetDFlashDraftModel"]

    def __init__(self, config) -> None:
        # Skip DFlashDraftModel.__init__: it builds a Qwen3 backbone. Everything
        # it sets up besides the layer stack is replicated below.
        Eagle3DraftModel.__init__(self, config)
        self.config = config

        dflash_config = dict(getattr(config, "dflash_config", None) or {})
        config.dflash_config = dflash_config

        register_target()
        self.target_config = _build_target_config(config)
        self.layers = nn.ModuleList(
            [
                TargetDSparkLayer(self.target_config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.target_layer_ids = dflash_config.get(
            "target_layer_ids"
        ) or build_target_layer_ids(
            config.num_target_layers, config.num_hidden_layers
        )
        self.fc = nn.Linear(
            len(self.target_layer_ids) * config.hidden_size,
            config.hidden_size,
            bias=False,
        )
        self.hidden_norm = TargetRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.norm = TargetRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.rotary_emb = TargetRotaryEmbedding(
            dim=config.qk_rope_head_dim,
            max_seq_len=config.max_position_embeddings,
            base=config.rope_theta,
        )
        self.block_size = config.block_size
        self.mask_token_id = dflash_config.get("mask_token_id", None)
        self.projector_type = dflash_config.get("projector_type", None)
        self.pure_draft_prefix_len = dflash_config.get("pure_draft_prefix_len", 0)
        self.shift_label = dflash_config.get("shift_label", False)
        self.anchor_prediction = dflash_config.get("anchor_prediction", False)
        self._init_draft_head(config, dflash_config)
        self.register_load_state_dict_pre_hook(self._remap_legacy_draft_head_keys)
        self.post_init()
        self._group_moe_experts_for_training()

    def _group_moe_experts_for_training(self) -> None:
        """Stack each layer's routed experts into trainable grouped weights.

        Same fix as ``target_dspark`` / ``target_mtp_eagle``: the acceptance
        reference dispatches the 256 routed experts from a Python loop, which
        dominates the step time. ``prepare_inference_weights`` stacks them into
        non-gradient buffers; re-register as Parameters so training optimises them.
        """
        for layer in self.layers:
            mlp = layer.mlp
            mlp.prepare_inference_weights()
            for name in ("_grouped_w13", "_grouped_w2"):
                stacked = getattr(mlp, name)
                del mlp._buffers[name]
                setattr(mlp, name, nn.Parameter(stacked))

    def forward(
        self,
        position_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        noise_embedding: Optional[torch.Tensor] = None,
        target_hidden: Optional[torch.Tensor] = None,
        past_key_values=None,
        use_cache: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        # Identical to TargetDSparkDraftModel.forward -- the backbone contract is
        # the same; only the (absent) head differs.
        del past_key_values, use_cache
        hidden_states = noise_embedding
        target_hidden = self.hidden_norm(self.fc(target_hidden))

        kv_position_ids = position_ids
        q_position_ids = position_ids[:, target_hidden.shape[1]:]

        for layer in self.layers:
            hidden_states = layer(
                hidden_states=hidden_states,
                target_hidden=target_hidden,
                attention_mask=attention_mask,
                q_position_ids=q_position_ids,
                kv_position_ids=kv_position_ids,
                rotary_emb=self.rotary_emb,
            )
        return self.norm(hidden_states)


__all__ = ["TargetDFlashDraftConfig", "TargetDFlashDraftModel"]
