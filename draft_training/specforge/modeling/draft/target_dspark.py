# coding=utf-8
"""target-layer-typed DSpark draft (DeepSpec per-family layer reuse).

DeepSpec implements DSpark per target family (`modeling/dspark/qwen3/`,
`gemma4/`, DSV4-style in serving): the draft backbone reuses the target
family's decoder layer type, and only the Markov/confidence heads plus the
loss are shared across families.  This module is the target family member of
that family:

- 5 x the target decoder layers (MLA attention + gated output + routed MoE FFN)
  with the DFlash dual-source key/value contract: queries come from the
  draft tokens, keys/values are the concatenation of the fc-projected
  target capture features (context part) and the draft hidden states
  (noise part), exactly like ``Qwen3DFlashAttention``'s k_ctx/k_noise.
- The routed experts are stacked into trainable grouped-GEMM parameters
  (same fix as ``target_mtp_eagle``: the acceptance reference dispatches
  256 experts from a Python loop).
- Markov/confidence heads are inherited from ``DSparkDraftModel``.  The
  upstream ``_init_draft_head`` only reads ``config.dflash_config`` while
  the trainers write ``config.dspark_config`` -- merge both dicts here so
  the heads are actually built (with the upstream flow they were silently
  dropped).

The target's MTP/decoder checkpoint weights are never loaded; every draft
parameter is cold-started.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.attention.flex_attention import BlockMask
from transformers import PretrainedConfig

from specforge.target import _ensure_acceptance_importable, register_target

from .base import Eagle3DraftModel
from .dflash import build_target_layer_ids
from .dspark import DSparkDraftModel
from .flex_attention import compile_friendly_flex_attention
from .registry import register_draft

_ensure_acceptance_importable()
from modeling import TargetConfig
from modeling.modeling import (
    TargetAttention,
    TargetMoE,
    TargetRMSNorm,
    TargetRotaryEmbedding,
    _apply_rotary,
)


class TargetDSparkDraftConfig(PretrainedConfig):
    model_type = "target_dspark_draft"

    def __init__(
        self,
        vocab_size: int = 201216,
        hidden_size: int = 1024,
        intermediate_size: int = 5120,
        num_hidden_layers: int = 5,
        num_attention_heads: int = 16,
        num_key_value_heads: int = 16,
        head_dim: int = 128,
        max_position_embeddings: int = 200192,
        q_lora_rank: int = 512,
        kv_lora_rank: int = 512,
        qk_rope_head_dim: int = 64,
        qk_nope_head_dim: int = 192,
        v_head_dim: int = 256,
        moe_intermediate_size: int = 1024,
        moe_latent_size: int = 512,
        n_routed_experts: int = 256,
        num_experts_per_tok: int = 8,
        n_shared_experts: int = 1,
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 10000.0,
        rope_interleave: bool = False,
        block_size: int = 16,
        num_target_layers: int = 24,
        dspark_config: Optional[dict] = None,
        dflash_config: Optional[dict] = None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_nope_head_dim = qk_nope_head_dim
        self.v_head_dim = v_head_dim
        self.moe_intermediate_size = moe_intermediate_size
        self.moe_latent_size = moe_latent_size
        self.n_routed_experts = n_routed_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.n_shared_experts = n_shared_experts
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.rope_interleave = rope_interleave
        self.block_size = block_size
        self.num_target_layers = num_target_layers
        self.dspark_config = dict(dspark_config or {})
        self.dflash_config = dict(dflash_config or {})
        super().__init__(**kwargs)


class TargetDSparkLayer(nn.Module):
    """One the target decoder layer under the DFlash dual-source attention."""

    def __init__(self, target_config: TargetConfig, layer_idx: int):
        super().__init__()
        self.self_attn = TargetAttention(target_config, layer_idx)
        self.mlp = TargetMoE(target_config)
        self.input_layernorm = TargetRMSNorm(
            target_config.hidden_size, target_config.rms_norm_eps
        )
        self.post_attention_layernorm = TargetRMSNorm(
            target_config.hidden_size, target_config.rms_norm_eps
        )
        self._target_config = target_config

    def forward(
        self,
        hidden_states: torch.Tensor,
        target_hidden: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        q_position_ids: torch.Tensor,
        kv_position_ids: torch.Tensor,
        rotary_emb: TargetRotaryEmbedding,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self._dual_source_attention(
            hidden_states,
            target_hidden,
            attention_mask,
            q_position_ids,
            kv_position_ids,
            rotary_emb,
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states

    def _dual_source_attention(
        self,
        hidden_states: torch.Tensor,
        target_hidden: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        q_position_ids: torch.Tensor,
        kv_position_ids: torch.Tensor,
        rotary_emb: TargetRotaryEmbedding,
    ) -> torch.Tensor:
        """MLA attention with DFlash k_ctx/k_noise semantics.

        Q comes from the draft tokens; K/V are projected from
        ``cat([target_hidden, hidden_states])`` with the shared kv
        projections, mirroring ``Qwen3DFlashAttention`` (k_ctx =
        k_proj(target), k_noise = k_proj(hidden)).  The attention core is
        SDPA with the DFlash boolean block mask so no [B, H, Q, KV] score
        matrix is materialised.
        """
        attn = self.self_attn
        cfg = self._target_config
        batch_size, q_len, _ = hidden_states.shape
        kv_source = torch.cat([target_hidden, hidden_states], dim=1)
        kv_len = kv_source.shape[1]

        # Q path (draft tokens only).
        q_latent = attn.q_a_layernorm(attn.q_a_proj(hidden_states))
        q_full = attn.q_b_proj(q_latent)
        q_full = q_full.view(
            batch_size, q_len, cfg.num_attention_heads,
            cfg.qk_nope_head_dim + cfg.qk_rope_head_dim,
        )
        q_nope = q_full[..., : cfg.qk_nope_head_dim]
        q_rope = q_full[..., cfg.qk_nope_head_dim:]

        # KV path over the dual source.
        kv_input = attn.kv_a_proj_with_mqa(kv_source)
        kv_latent = kv_input[..., : cfg.kv_lora_rank]
        k_rope = kv_input[..., cfg.kv_lora_rank:]
        kv_latent = attn.kv_a_layernorm(kv_latent)
        kv_full = attn.kv_b_proj(kv_latent)
        kv_full = kv_full.view(
            batch_size, kv_len, cfg.num_attention_heads,
            cfg.qk_nope_head_dim + cfg.v_head_dim,
        )
        k_nope = kv_full[..., : cfg.qk_nope_head_dim]
        v = kv_full[..., cfg.qk_nope_head_dim:]

        # RoPE: query positions are the draft (anchor + offset) positions;
        # key positions cover the context part first, then the same draft
        # positions -- i.e. the caller's full position ids.  Both id tensors
        # are [B, L]; index_select flattens them, so reshape the selected
        # rows back to [B, L, 1, D] to broadcast against [B, L, H, D]
        # (mirrors modeling._select_rotary; the old [1, B*L, 1, D]
        # layout only worked for B == 1).
        q_positions = q_position_ids.reshape(-1).to(torch.long)
        kv_positions = kv_position_ids.reshape(-1).to(torch.long)
        max_position = int(
            torch.maximum(
                q_positions.max().unsqueeze(0),
                kv_positions.max().unsqueeze(0),
            ).item()
        ) + 1
        cos, sin = rotary_emb(max_position, hidden_states.device)
        cos_q = cos.index_select(0, q_positions).reshape(batch_size, q_len, -1).unsqueeze(2)
        sin_q = sin.index_select(0, q_positions).reshape(batch_size, q_len, -1).unsqueeze(2)
        q_rope = _apply_rotary(q_rope, cos_q, sin_q, cfg.rope_interleave)
        cos_k = cos.index_select(0, kv_positions).reshape(batch_size, kv_len, -1).unsqueeze(2)
        sin_k = sin.index_select(0, kv_positions).reshape(batch_size, kv_len, -1).unsqueeze(2)
        k_rope = k_rope.unsqueeze(2).expand(-1, -1, cfg.num_attention_heads, -1)
        k_rope = _apply_rotary(k_rope, cos_k, sin_k, cfg.rope_interleave)

        query = torch.cat([q_nope, q_rope], dim=-1).transpose(1, 2)
        key = torch.cat([k_nope, k_rope], dim=-1).transpose(1, 2)
        value = v.transpose(1, 2)

        if attention_mask is not None and isinstance(attention_mask, BlockMask):
            # flex_attention backend: the trainer hands us a BlockMask, which
            # SDPA refuses ("attn_mask must be Tensor, not BlockMask").  Route
            # through the compiled flex wrapper the qwen3 path reaches via
            # transformers' attention registry.
            attn_out = compile_friendly_flex_attention(
                query,
                key,
                value,
                block_mask=attention_mask,
                scale=attn.softmax_scale,
            )
        else:
            attn_out = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=attention_mask,
                scale=attn.softmax_scale,
            )
        attn_out = attn_out.transpose(1, 2).contiguous().view(
            batch_size, q_len, cfg.num_attention_heads * cfg.v_head_dim
        )
        if attn.use_gated_attn:
            attn_out = attn_out * torch.sigmoid(attn.gate_proj(hidden_states))
        return attn.o_proj(attn_out)


@register_draft
class TargetDSparkDraftModel(DSparkDraftModel):
    """DSpark draft with a target-5 decoder backbone and the DSpark heads."""

    config_class = TargetDSparkDraftConfig
    _no_split_modules = ["TargetDSparkLayer"]
    architectures = ["TargetDSparkDraftModel"]

    def __init__(self, config) -> None:
        # Skip DFlashDraftModel.__init__ (it builds a Qwen3 backbone).  The
        # heads are initialised manually below with a merged config dict so
        # that markov/confidence settings written to config.dspark_config by
        # the trainer are not silently dropped.
        Eagle3DraftModel.__init__(self, config)
        self.config = config

        dspark_config = dict(getattr(config, "dspark_config", None) or {})
        dflash_config = dict(getattr(config, "dflash_config", None) or {})
        dflash_config.setdefault("projector_type", self.expected_projector_type)
        config.dspark_config = dspark_config
        config.dflash_config = dflash_config

        register_target()
        target_config = TargetConfig(
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
        self.target_config = target_config

        self.layers = nn.ModuleList(
            [
                TargetDSparkLayer(target_config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.target_layer_ids = dspark_config.get(
            "target_layer_ids"
        ) or dflash_config.get(
            "target_layer_ids"
        ) or build_target_layer_ids(
            config.num_target_layers, config.num_hidden_layers
        )
        self.fc = nn.Linear(
            len(self.target_layer_ids) * config.hidden_size,
            config.hidden_size,
            bias=False,
        )
        self.hidden_norm = TargetRMSNorm(
            config.hidden_size, config.rms_norm_eps
        )
        self.norm = TargetRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.rotary_emb = TargetRotaryEmbedding(
            dim=config.qk_rope_head_dim,
            max_seq_len=config.max_position_embeddings,
            base=config.rope_theta,
        )
        self.block_size = config.block_size
        self.mask_token_id = dspark_config.get(
            "mask_token_id", None
        ) or dflash_config.get("mask_token_id", None)

        self._init_draft_head(
            config, {**dspark_config, **dflash_config}
        )
        self.post_init()
        self._group_moe_experts_for_training()

    def _group_moe_experts_for_training(self) -> None:
        """Stack each layer's routed experts into trainable grouped weights.

        Same fix as ``target_mtp_eagle``: the acceptance reference dispatches
        the 256 routed experts from a Python loop, which dominates the step
        time when it runs for every layer.  ``prepare_inference_weights``
        stacks the experts into non-gradient buffers; re-register them as
        Parameters so training keeps them optimised.
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
        del past_key_values, use_cache
        hidden_states = noise_embedding
        target_hidden = self.hidden_norm(self.fc(target_hidden))

        # position_ids layout (OnlineDSparkModel): [context 0..S-1] then the
        # draft block positions (anchor + offset).  Queries are the noise
        # tokens; keys/values span both parts.
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


__all__ = ["TargetDSparkDraftConfig", "TargetDSparkDraftModel"]
