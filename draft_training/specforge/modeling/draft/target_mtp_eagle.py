# coding=utf-8
"""Cold-start EAGLE drafter with the target decoder/MTP layer geometry."""

from __future__ import annotations

import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.configuration_utils import PretrainedConfig

from specforge.target import _ensure_acceptance_importable, register_target

from .base import Eagle3DraftModel
from .registry import register_draft


class TargetMTPEagleConfig(PretrainedConfig):
    model_type = "target_mtp_eagle"

    def __init__(
        self,
        vocab_size: int = 201216,
        draft_vocab_size: int = 32000,
        hidden_size: int = 1024,
        target_hidden_size: int = 1024,
        num_aux_hidden_states: int = 3,
        num_hidden_layers: int = 1,
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
        route_layer_ids=None,
        initializer_range: float = 0.02,
        pad_token_id: int = 2,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.draft_vocab_size = draft_vocab_size
        self.hidden_size = hidden_size
        self.target_hidden_size = target_hidden_size
        self.num_aux_hidden_states = num_aux_hidden_states
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
        self.route_layer_ids = list(route_layer_ids or [1, 11, 20])
        self.initializer_range = initializer_range
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=False,
            **{key: value for key, value in kwargs.items()
               if key != "tie_word_embeddings"},
        )


@register_draft
class TargetMTPEagleModel(Eagle3DraftModel):
    """the target-shaped, cold-start EAGLE draft model.

    The target's MTP/decoder parameters are intentionally never loaded. This
    class only mirrors the target layer contract so that cold-start training
    compares the EAGLE objective without mixing in a warm-start advantage.
    """

    config_class = TargetMTPEagleConfig

    def __init__(self, config, quant_config=None, attention_backend="sdpa"):
        del quant_config
        super().__init__(config)
        self.config = config
        self.attention_backend = attention_backend
        self.vocab_size = config.vocab_size
        self.draft_vocab_size = config.draft_vocab_size

        register_target()
        _ensure_acceptance_importable()
        from modeling import (
            TargetConfig,
            TargetMTPHead,
        )
        from modeling.modeling import (
            TargetRotaryEmbedding,
            _apply_rotary,
        )

        target_config = TargetConfig(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            num_hidden_layers=24,
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
            rope_interleave=False,
            first_k_dense_replace=0,
            moe_layer_freq=1,
            rms_norm_eps=config.rms_norm_eps,
            rope_theta=config.rope_theta,
            use_mtp=True,
            num_nextn_predict_layers=1,
            mtp_reuse_last_layer=False,
        )
        self.target_config = target_config
        self._apply_rotary = _apply_rotary
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, config.pad_token_id
        )
        self.rotary_emb = TargetRotaryEmbedding(
            dim=config.qk_rope_head_dim,
            max_seq_len=config.max_position_embeddings,
            base=config.rope_theta,
        )
        # Independent MTP layer: same structure as model.layers[24], random
        # initialized, with no checkpoint weights copied into it.
        self.mtp_head = TargetMTPHead(target_config, shared_layer=None)
        self.fc = nn.Linear(
            config.target_hidden_size * config.num_aux_hidden_states,
            config.hidden_size,
            bias=False,
        )
        self.lm_head = nn.Linear(
            config.hidden_size, config.draft_vocab_size, bias=False
        )
        self.register_buffer(
            "t2d", torch.ones(config.vocab_size, dtype=torch.bool)
        )
        self.register_buffer(
            "d2t", torch.zeros(config.draft_vocab_size, dtype=torch.long)
        )
        self.post_init()
        self._group_mtp_experts_for_training()

    def _group_mtp_experts_for_training(self) -> None:
        """Pack the MTP head's routed experts into trainable stacked weights.

        The acceptance-package reference dispatches the 256 routed experts from
        a Python loop (mask + small GEMM per expert). The TTT loop calls the
        draft backbone seven times per training step, forward and backward, so
        that loop would dominate the step time -- the same pathology DSV4 hit
        and fixed with stacked experts (DSV4_MTP_WARMSTART_PERF root cause 2).

        ``prepare_inference_weights()`` already stacks the expert tensors and
        drops the per-expert modules, but parks the stacks in non-gradient
        buffers (an inference-only layout). Re-register them as Parameters so
        autograd, FSDP and the optimizer keep training them.
        """
        mlp = self.mtp_head.transformer_layer.mlp
        mlp.prepare_inference_weights()
        for name in ("_grouped_w13", "_grouped_w2"):
            stacked = getattr(mlp, name)
            del mlp._buffers[name]
            setattr(mlp, name, nn.Parameter(stacked))

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def project_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.shape[-1] != self.fc.in_features:
            raise ValueError(
                f"Expected {self.fc.in_features} concatenated target features, "
                f"got {hidden_states.shape[-1]}"
            )
        return self.fc(hidden_states)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states)

    def _recurrent_attention(
        self,
        hidden: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_ids: torch.Tensor,
        cache_hidden,
    ) -> torch.Tensor:
        """Run the target MLA with EAGLE's per-position depth cache.

        EAGLE's ``cache_hidden`` stores one K/V tensor per TTT depth.  The first
        depth attends causally over the whole sequence; each later depth adds one
        same-position key/value per token.  Concatenating these tensors on the
        sequence axis would change the objective and does not match the existing
        Llama EAGLE implementation.
        """
        attn = self.mtp_head.transformer_layer.self_attn
        batch_size, seq_length, _ = hidden.shape
        depth = len(cache_hidden[0]) if cache_hidden is not None else 0

        q_latent = attn.q_a_layernorm(attn.q_a_proj(hidden))
        q_full = attn.q_b_proj(q_latent).view(
            batch_size,
            seq_length,
            attn.num_heads,
            attn.qk_nope_head_dim + attn.qk_rope_head_dim,
        )
        kv_input = attn.kv_a_proj_with_mqa(hidden)
        kv_latent = attn.kv_a_layernorm(kv_input[..., :attn.kv_lora_rank])
        kv_full = attn.kv_b_proj(kv_latent).view(
            batch_size,
            seq_length,
            attn.num_heads,
            attn.qk_nope_head_dim + attn.v_head_dim,
        )

        q_nope, q_rope = q_full.split(
            [attn.qk_nope_head_dim, attn.qk_rope_head_dim], dim=-1
        )
        k_nope, value = kv_full.split(
            [attn.qk_nope_head_dim, attn.v_head_dim], dim=-1
        )

        position_ids = position_ids.to(device=hidden.device, dtype=torch.long).reshape(-1)
        if position_ids.numel() != seq_length:
            raise ValueError(
                f"Expected {seq_length} query positions, got {position_ids.numel()}"
            )
        position_ids = position_ids + depth
        max_position = int(position_ids.max().item()) + 1
        cos, sin = self.rotary_emb(max_position, hidden.device)
        cos = cos.index_select(0, position_ids).unsqueeze(0).unsqueeze(2)
        sin = sin.index_select(0, position_ids).unsqueeze(0).unsqueeze(2)
        q_rope = self._apply_rotary(q_rope, cos, sin, attn.rope_interleave)
        # The MLA KV projection carries a single shared RoPE slice per token.
        kv_rope = kv_input[..., attn.kv_lora_rank:]
        kv_rope = kv_rope.unsqueeze(2).expand(
            -1, -1, attn.num_heads, -1
        )
        k_rope = self._apply_rotary(kv_rope, cos, sin, attn.rope_interleave)
        key = torch.cat([k_nope, k_rope], dim=-1).transpose(1, 2)
        query = torch.cat([q_nope, q_rope], dim=-1).transpose(1, 2)
        value = value.transpose(1, 2)

        if cache_hidden is None:
            scores = torch.matmul(query, key.transpose(-2, -1)) * attn.softmax_scale
            if attention_mask is not None:
                scores = scores + attention_mask
            weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
            attended = torch.matmul(weights, value)
        else:
            cache_hidden[0].append(key)
            cache_hidden[1].append(value)
            if os.environ.get("TARGET_EAGLE_TTT_EAGER", "0") != "1":
                attended = self._flex_ttt_attention(
                    query, value, cache_hidden, attention_mask, attn
                )
            else:
                attended = self._eager_ttt_attention(
                    query, value, cache_hidden, attention_mask, attn
                )

        attended = attended.transpose(1, 2).contiguous().view(
            batch_size, seq_length, attn.num_heads * attn.v_head_dim
        )
        if attn.use_gated_attn:
            attended = attended * torch.sigmoid(attn.gate_proj(hidden))
        return attn.o_proj(attended)

    def _flex_ttt_attention(self, query, value, cache_hidden, attention_mask, attn):
        """Depth-chained TTT attention via flex_attention.

        Mirrors ``LlamaFlexAttention`` (llama3_eagle.py): the KV sequence is
        [depth-0 block | depth-1 block | ... | current block]; queries attend
        causally over the depth-0 block and to the same-position (diagonal)
        entry of every later block -- exactly the eager path's base_key +
        per-depth elementwise-dot semantics, expressed through
        ``generate_eagle3_mask``.  flex_attention never materialises the
        [B, H, S, KV] score matrix, removing the O(S^2) memory (the B=2 OOM
        cause) and the per-step fp32 softmax cost.

        Set TARGET_EAGLE_TTT_EAGER=1 to fall back to the eager path.
        """
        from .flex_attention import (
            compile_friendly_create_block_mask,
            compile_friendly_flex_attention,
            generate_eagle3_mask,
        )

        keys = torch.cat(cache_hidden[0], dim=2)
        values = torch.cat(cache_hidden[1], dim=2)
        batch_size, _, seq_len, _ = query.shape
        kv_len = keys.shape[2]

        # True (unpadded) length per row: the last causal query row allows a
        # key iff that key position is real, so count its zero entries.  B=1
        # with no padding collapses to seq_len.
        if attention_mask is not None and attention_mask.dim() == 4:
            seq_lengths = (
                (attention_mask[:, 0, -1, :] == 0).sum(dim=-1).to(torch.int32)
            )
        else:
            seq_lengths = torch.full(
                (batch_size,), seq_len, device=query.device, dtype=torch.int32
            )

        mask_mod = generate_eagle3_mask(seq_lengths, seq_len, kv_len)
        block_mask = compile_friendly_create_block_mask(
            mask_mod,
            B=batch_size,
            H=1,
            Q_LEN=seq_len,
            KV_LEN=kv_len,
            device=query.device,
        )
        attended = compile_friendly_flex_attention(
            query=query,
            key=keys,
            value=values,
            block_mask=block_mask,
            scale=attn.softmax_scale,
        )
        # flex returns [B, H, S, v_head_dim]; match the eager path's layout.
        return attended

    def _eager_ttt_attention(self, query, value, cache_hidden, attention_mask, attn):
        """Original reference path: explicit scores + fp32 softmax."""
        base_key = cache_hidden[0][0]
        base_value = cache_hidden[1][0]
        scores = torch.matmul(query, base_key.transpose(-2, -1)) * attn.softmax_scale
        if attention_mask is not None:
            scores = scores + attention_mask
        for cached_key in cache_hidden[0][1:]:
            extra = (query * cached_key).sum(dim=-1) * attn.softmax_scale
            scores = torch.cat([scores, extra.unsqueeze(-1)], dim=-1)
        weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
        attended = torch.matmul(weights[..., : base_key.shape[2]], base_value)
        for index, cached_value in enumerate(cache_hidden[1][1:], start=base_key.shape[2]):
            attended = attended + weights[..., index:index + 1] * cached_value
        return attended

    def backbone(
        self,
        input_embeds: torch.Tensor,
        hidden_states: torch.Tensor,
        cache_hidden: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values=None,
        use_cache: bool = True,
    ) -> torch.Tensor:
        del past_key_values, use_cache
        if position_ids.ndim > 1:
            position_ids = position_ids[0]
        position_ids = position_ids.to(device=hidden_states.device, dtype=torch.long)
        mtp = self.mtp_head
        fused = mtp.eh_proj(
            torch.cat([mtp.enorm(input_embeds), mtp.hnorm(hidden_states)], dim=-1)
        )
        layer = mtp.transformer_layer
        residual = fused
        hidden = layer.input_layernorm(fused)
        hidden = residual + self._recurrent_attention(
            hidden,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache_hidden=cache_hidden,
        )
        hidden = hidden + layer.mlp(layer.post_attention_layernorm(hidden))
        return mtp.shared_head["norm"](hidden)
