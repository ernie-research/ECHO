"""the target V2 PyTorch HuggingFace modeling.

Architecture: 10B-A1B (10B total, 1B activated) Latent-MoE + MLA + ECHO-MTP.
Derived from DeepSeek-V3 with the following differences:
  1. scoring_func="sigmoid"  — sigmoid-based MoE routing (not softmax)
  2. use_gated_attn=True      — additional gate_proj inside self-attention
  3. routed_scaling_factor_learnable=True — scaling param in state_dict
  4. MTP head: mtp_reuse_last_layer=True  (ECHO) reuses layer L-1 weights
               mtp_reuse_last_layer=False (DSpark/DeepSeek) independent MTP layer
  5. Weight key names from checkpoint (already HF-style, no rename needed):
       model.layers.N.self_attn.{q_a_proj, q_b_proj, kv_a_proj_with_mqa, kv_b_proj, o_proj, gate_proj}
       model.layers.N.block_sparse_moe.{gate, experts.N.{w1,w2,w3}, shared_experts.{w1,w2,w3}}
       model.layers.N.{input_layernorm, post_attention_layernorm}
       model.layers.24.{enorm, hnorm, eh_proj, shared_head.norm}
       model.embed_tokens.weight  /  model.norm.weight  /  lm_head.weight

Weight shapes confirmed from checkpoint (all [out_features, in_features]):
  q_a_proj:  [512, 1024]    (q_lora_rank, hidden)
  q_b_proj:  [4096, 512]    (n_heads*(qk_nope+qk_rope)=16*256, q_lora_rank)
  kv_a_proj: [576, 1024]    (kv_lora_rank+qk_rope_head_dim, hidden)
  kv_b_proj: [7168, 512]    (n_heads*(qk_nope+v_head_dim)=16*448, kv_lora_rank)
  o_proj:    [1024, 4096]   (hidden, n_heads*v_head_dim=16*256)
  gate_proj: [4096, 1024]   (n_heads*v_head_dim, hidden) — input is hidden, not v_out
  MoE gate.weight:  [256, 1024]   (n_experts, hidden) — direct routing from hidden
  MoE fc1_latent:  [512, 1024]   (moe_latent_size, hidden)
  MoE fc2_latent:  [1024, 512]   (hidden, moe_latent_size)
  MoE experts w1/w3: [1024, 512] each (moe_intermediate_size, moe_latent_size)
  MoE experts w2:    [512, 1024]  (moe_latent_size, moe_intermediate_size)
  shared w1/w3:  [1024, 1024]    (moe_intermediate_size, hidden)
  shared w2:     [1024, 1024]    (hidden, moe_intermediate_size)
  eh_proj (MTP): [1024, 2048]    (hidden, 2*hidden)
"""
from __future__ import annotations

import math
import os
from typing import Optional, Tuple, Union, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GenerationMixin, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast
from transformers.cache_utils import Cache, DynamicCache

from .configuration import TargetConfig

# ----------------------------------------------------------------------------
# Attention implementation switch (env-gated; default preserves the original path)
#
#   TARGET_ATTN = eager        explicit matmul + fp32 softmax (original reference path)
#                 | sdpa_flash   SDPA pinned to the FlashAttention backend (FA2-class) [DEFAULT]
#                 | sdpa_cudnn   SDPA pinned to the cuDNN fused backend — on Hopper this is
#                                the FA3-class warp-specialised kernel, i.e. the closest
#                                match to PaddleFleet's `fa_version: 3` + flashmask config
#                 | sdpa_mem     SDPA pinned to the memory-efficient backend
#                 | sdpa_math    SDPA pinned to the math backend
#                 | sdpa         SDPA with PyTorch's own backend choice
#
# Measured shape-sensitivity of the main logits (12-token probe, length 12 vs 16):
#   eager 0.976562 (argmax 11/11) | sdpa_flash 0.562500 (11/11) | sdpa_cudnn 0.562500 (10/11)
#   Paddle, same test: 0.000000.  `A vs A2` at a fixed shape is 0.000000 for all of them.
# sdpa_flash is the DEFAULT: it halves the shape-dependence (0.562500 vs eager's 0.976562) while
# keeping argmax stable across shapes, unlike sdpa_cudnn.  Results recorded before 2026-08-03 were
# produced with `eager`; set TARGET_ATTN=eager to reproduce those exactly.
# Matching Paddle's `fa_version: 3` more closely (sdpa_cudnn) buys nothing measurable, because the
# residual is dominated by the MoE expert GEMMs (L0.mlp = 0.003906 under BOTH backends), not by
# attention -- and cuDNN is the only backend whose own argmax moves with shape (10/11).
#
# Why this exists: `hf_shape_bisect.py` localised HF's SHAPE-DEPENDENCE to
# `L0.self_attn` — eager `matmul(q, k^T)` over the full [T,T] score matrix lets cuBLAS
# pick different tiling/split-K per sequence length, so rows 0..m-1 move by up to ~1 ulp
# when trailing tokens are added (measured: main-logit max|diff| 0.976562 between length
# 12 and 16, while `A vs A2` at a fixed shape is 0.000000).  Paddle's fused flashmask path
# measures exactly 0.000000 on the same test, because flash accumulates causally over
# fixed-size KV blocks: row i never sees blocks beyond i, so total length cannot change it.
#
# Shape-dependence is a real defect for a reference implementation: within HF, the
# target-only decode loop and the MTP verifier evaluate the same position at DIFFERENT
# sequence lengths, so a thin-margin argmax can flip between the two paths (observed:
# target_ids == mtp_ids on only 32/50 greedy questions for HF vs 38/50 for Paddle).
# ----------------------------------------------------------------------------
ATTN_IMPL = os.environ.get("TARGET_ATTN", "sdpa_flash").lower()
# Accumulate the MoE expert combine in fp32 (see TargetMoE.forward). Off by default.
MOE_FP32_ACC = os.environ.get("TARGET_MOE_FP32ACC", "0") == "1"
# Re-break FIX #4 on purpose (measurement only): evaluate MoE routing in bf16 as the original
# port did. Used to quantify how much of the layer-by-layer misalignment that one bug caused.
GATE_BF16 = os.environ.get("TARGET_GATE_BF16", "0") == "1"
_SDPA_BACKENDS = {
    "sdpa_flash": "FLASH_ATTENTION",
    "sdpa_cudnn": "CUDNN_ATTENTION",
    "sdpa_mem": "EFFICIENT_ATTENTION",
    "sdpa_math": "MATH",
}


def _sdpa_ctx():
    """Context manager pinning the SDPA backend, or a no-op for plain `sdpa`/eager."""
    import contextlib
    name = _SDPA_BACKENDS.get(ATTN_IMPL)
    if name is None:
        return contextlib.nullcontext()
    from torch.nn.attention import SDPBackend, sdpa_kernel
    return sdpa_kernel(getattr(SDPBackend, name))


def _is_pure_causal(mask: Optional[torch.Tensor], q_len: int, kv_len: int) -> bool:
    """True when `mask` is exactly an additive upper-triangular causal mask.

    Only then can SDPA be called with is_causal=True (which is what keeps it on the
    flash kernel); any other mask forces the eager fallback.

    "Masked" is accepted as either -inf OR a very large negative value: callers legitimately
    use `torch.finfo(dtype).min` instead of -inf (e.g. paired_hitrate_hf.py), and treating that
    as "not causal" silently routed those runs to PyTorch's backend auto-selection, which picked
    cuDNN and blew up with `mha_graph->execute(...) is_good() == false`.
    """
    if mask is None:
        return True
    if q_len != kv_len or mask.shape[-2:] != (q_len, kv_len):
        return False
    lower = torch.ones(q_len, kv_len, dtype=torch.bool, device=mask.device).tril()
    thresh = torch.finfo(mask.dtype).min / 2
    masked = (mask <= thresh) | torch.isneginf(mask)      # -inf or ~finfo.min
    keep = ~masked
    return bool((keep == lower).all()) and bool((mask[keep] == 0).all())




# ============================================================================
# Basic building blocks
# ============================================================================

class TargetRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        var = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.variance_epsilon)
        return (self.weight * x).to(dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def _apply_rotary(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    interleaved: bool,
) -> torch.Tensor:
    if not interleaved:
        # PaddleFleet's MLA path groups the even and odd projection dimensions
        # before applying the standard split-half rotation. The grouped layout
        # is retained in the final Q/K tensors.
        x = torch.cat([x[..., 0::2], x[..., 1::2]], dim=-1)
        return (x * cos) + (_rotate_half(x) * sin)

    # PaddleFleet's EC-compatible RoPE treats adjacent values as complex pairs.
    # cos/sin are stored split-half, so use their first half as per-pair phases.
    half = x.shape[-1] // 2
    cos_pairs = cos[..., :half]
    sin_pairs = sin[..., :half]
    real = x[..., 0::2]
    imag = x[..., 1::2]
    return torch.stack(
        [real * cos_pairs - imag * sin_pairs, real * sin_pairs + imag * cos_pairs],
        dim=-1,
    ).flatten(-2)


def _normalize_position_ids(
    position_ids: Optional[torch.Tensor],
    batch_size: int,
    sequence_length: int,
    device: torch.device,
    name: str,
) -> torch.Tensor:
    if position_ids is None:
        return torch.arange(sequence_length, device=device, dtype=torch.long)
    position_ids = position_ids.to(device=device, dtype=torch.long)
    if position_ids.ndim == 1 and position_ids.numel() == sequence_length:
        return position_ids
    if position_ids.ndim == 2 and position_ids.shape == (
        batch_size,
        sequence_length,
    ):
        return position_ids
    raise ValueError(
        f"{name} must have shape ({sequence_length},) or "
        f"({batch_size}, {sequence_length}), got {tuple(position_ids.shape)}"
    )


def _select_rotary(
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor,
    batch_size: int,
    sequence_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    selected_cos = cos.index_select(0, position_ids.reshape(-1))
    selected_sin = sin.index_select(0, position_ids.reshape(-1))
    if position_ids.ndim == 1:
        return (
            selected_cos.unsqueeze(0).unsqueeze(2),
            selected_sin.unsqueeze(0).unsqueeze(2),
        )
    return (
        selected_cos.reshape(batch_size, sequence_length, -1).unsqueeze(2),
        selected_sin.reshape(batch_size, sequence_length, -1).unsqueeze(2),
    )


def apply_rotary_emb(q: torch.Tensor, k: torch.Tensor,
                     cos: torch.Tensor, sin: torch.Tensor,
                     interleaved: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    return (
        _apply_rotary(q, cos, sin, interleaved),
        _apply_rotary(k, cos, sin, interleaved),
    )


class TargetRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 200192, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        # Kept for state-dict/introspection compatibility only -- forward() never reads it,
        # because `.to(dtype=bf16)` casts float buffers and would quantise the frequencies.
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.dim = dim
        self.base = base
        self.max_seq_len = max_seq_len

    @torch.no_grad()
    def forward(self, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        # BOTH the position ramp and the frequencies must be fp32, and the frequencies must be
        # RECOMPUTED rather than upcast: `.to(dtype=bf16)` (torch_weight_loader.py:179) casts
        # float buffers, so `self.inv_freq` is bf16 and `.float()` only widens already-rounded
        # values. Two separate errors came from that single cast:
        #   * positions: bf16 spaces integers by 2 above 256 and by 16 above 2048 (max |pos
        #     error| 8, 3900 -> 3896), which destroyed RoPE's relative property -- the same 512
        #     tokens scored 2.0379 at positions 0..511 but 3.9032 at 3584..4095.
        #   * frequencies: inv_freq is ~1e-4, bf16 carries ~0.4% relative error, i.e. an
        #     effective rope base mismatch vs the trained weights. This one is invisible to a
        #     position-shift test (q and k share the same wrong frequencies, so the dot product
        #     still depends only on the relative distance) but is a real absolute error.
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2, device=device, dtype=torch.float32) / self.dim)
        )
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        out_dtype = self.inv_freq.dtype
        return emb.cos().to(out_dtype), emb.sin().to(out_dtype)


# ============================================================================
# MoE Router (sigmoid-based, the target style)
# ============================================================================

class TargetMoEGate(nn.Module):
    """Sigmoid-based MoE router — routes directly from hidden_size (not moe_latent_size).
    Confirmed from ckpt: gate.weight shape = [256, 1024] (n_experts, hidden_size).
    """

    def __init__(self, config: TargetConfig):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts
        self.scoring_func = config.scoring_func
        self.routed_scaling_factor_learnable = config.routed_scaling_factor_learnable

        # Routes from hidden_size directly (confirmed: [256, 1024])
        self.weight = nn.Parameter(
            torch.empty(config.n_routed_experts, config.hidden_size)
        )
        if config.routed_scaling_factor_learnable:
            self.routed_scaling_factor_param = nn.Parameter(
                torch.full((config.n_routed_experts,), config.routed_scaling_factor)
            )
        else:
            self.register_buffer(
                "routed_scaling_factor_param",
                torch.tensor(config.routed_scaling_factor),
            )

        self.register_buffer(
            "e_score_correction_bias",
            torch.zeros(config.n_routed_experts),
            persistent=True,
        )
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(
        self,
        hidden: torch.Tensor,
        correction_bias: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # hidden: [B*T, hidden_size]
        # Routing MUST be evaluated in fp32.  Sigmoid scores are O(1e-4..1e-1) while the
        # correction bias is ~0.58, and bf16 resolution near 0.58 is ~0.0078 - so a bf16
        # `scores + bias` quantises small scores away entirely and corrupts top-k
        # selection.  Paddle evaluates routing in fp32: its actual expert selection is
        # reproduced EXACTLY (11/11 tokens) by an fp64 recomputation of
        # top_k(sigmoid(x @ w.T) + bias), whereas bf16 variants match only ~5/11.
        if GATE_BF16:
            # Reproduces FIX #4 in its BROKEN state, for measurement only (TARGET_GATE_BF16=1).
            # Whole routing path in bf16: `scores + bias` then quantises small scores away.
            logits = F.linear(hidden, self.weight)
            scores = logits.sigmoid() if self.scoring_func == "sigmoid" else logits.softmax(dim=-1)
            bias = self.e_score_correction_bias if correction_bias is None else correction_bias
            routing_scores = scores + bias.unsqueeze(0)
            topk_indices = torch.topk(routing_scores, self.top_k, dim=-1).indices
            topk_weights = scores.gather(1, topk_indices)
            topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-9)
            if self.routed_scaling_factor_learnable:
                sel = self.routed_scaling_factor_param.gather(0, topk_indices.reshape(-1))
                topk_weights = topk_weights * sel.view_as(topk_indices)
            else:
                topk_weights = topk_weights * self.routed_scaling_factor_param
            return topk_indices, topk_weights.to(hidden.dtype)

        logits = F.linear(hidden.float(), self.weight.float())  # [B*T, n_experts], fp32

        if self.scoring_func == "sigmoid":
            scores = logits.sigmoid()
        else:
            scores = logits.softmax(dim=-1)

        bias = self.e_score_correction_bias if correction_bias is None else correction_bias
        routing_scores = scores + bias.float().unsqueeze(0)
        topk_indices = torch.topk(routing_scores, self.top_k, dim=-1).indices
        topk_weights = scores.gather(1, topk_indices)
        topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-9)

        if self.routed_scaling_factor_learnable:
            selected_scales = self.routed_scaling_factor_param.float().gather(
                0, topk_indices.reshape(-1)
            )
            topk_weights = topk_weights * selected_scales.view_as(topk_indices)
        else:
            topk_weights = topk_weights * self.routed_scaling_factor_param.float()
        return topk_indices, topk_weights.to(hidden.dtype)


# ============================================================================
# MoE FFN
# ============================================================================

class TargetExpert(nn.Module):
    """Single routed expert: latent_in → up/gate → act → down → latent_out."""

    def __init__(self, config: TargetConfig):
        super().__init__()
        # w1 (up) and w3 (gate) are stored separately in checkpoint but fused here
        # as up_gate_proj: [moe_latent_size, 2 * moe_intermediate_size]
        self.up_gate_proj = nn.Linear(
            config.moe_latent_size, 2 * config.moe_intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            config.moe_intermediate_size, config.moe_latent_size, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, value = self.up_gate_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * value)


class TargetSharedExpert(nn.Module):
    """Shared expert: hidden_size → up/gate → act → down → hidden_size."""

    def __init__(self, config: TargetConfig):
        super().__init__()
        self.up_gate_proj = nn.Linear(
            config.hidden_size, 2 * config.moe_intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            config.moe_intermediate_size, config.hidden_size, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, value = self.up_gate_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * value)


class TargetMoE(nn.Module):
    """Block-sparse MoE: hidden → gate (routing) + latent_proj + experts + shared expert.

    Architecture (from ckpt shapes):
      gate.weight:     [n_experts=256, hidden=1024]  — routes from hidden directly
      fc1_latent_proj: [moe_latent_size=512, hidden=1024]  — [out, in] = Linear(1024, 512)
      fc2_latent_proj: [hidden=1024, moe_latent_size=512]  — [out, in] = Linear(512, 1024)
      experts.N.w1/w3: [moe_ffn=1024, moe_latent=512]     — expert takes latent input
      experts.N.w2:    [moe_latent=512, moe_ffn=1024]
      shared w1/w3:    [moe_ffn=1024, hidden=1024]         — shared expert takes hidden input
      shared w2:       [hidden=1024, moe_ffn=1024]
    """

    def __init__(self, config: TargetConfig):
        super().__init__()
        # Latent projections for expert computation (hidden → latent → hidden)
        self.fc1_latent_proj = nn.Linear(config.hidden_size, config.moe_latent_size, bias=False)
        self.fc2_latent_proj = nn.Linear(config.moe_latent_size, config.hidden_size, bias=False)
        # Gate routes from hidden directly
        self.gate = TargetMoEGate(config)
        self.experts = nn.ModuleList([
            TargetExpert(config) for _ in range(config.n_routed_experts)
        ])
        self.shared_experts = TargetSharedExpert(config)
        self.register_buffer("_grouped_w13", None, persistent=False)
        self.register_buffer("_grouped_w2", None, persistent=False)

    def prepare_inference_weights(self) -> None:
        """Pack routed experts for grouped GEMM inference.

        The reference implementation keeps one Module per expert and launches
        separate small GEMMs from a Python loop. That is unusable for
        autoregressive decode with 256 experts. Packing after checkpoint load
        preserves the loaded weights while replacing that dispatch path with
        grouped BF16 GEMMs.
        """
        if self._grouped_w13 is not None:
            return
        if not hasattr(torch, "_grouped_mm"):
            raise RuntimeError("PyTorch grouped_mm is required for inference packing")
        if self.experts is None:
            raise RuntimeError("routed experts are already packed")

        self._grouped_w13 = torch.stack(
            [expert.up_gate_proj.weight for expert in self.experts],
            dim=0,
        ).contiguous()
        self._grouped_w2 = torch.stack(
            [expert.down_proj.weight for expert in self.experts],
            dim=0,
        ).contiguous()
        # The packed tensors now own the inference copy; drop the per-expert
        # modules so packing does not increase the model's resident weight size.
        self.experts = None

    def _forward_grouped(
        self,
        latent: torch.Tensor,
        topk_indices: torch.Tensor,
        topk_weights: torch.Tensor,
        acc_dtype: torch.dtype,
    ) -> torch.Tensor:
        """Run routed experts with grouped BF16 GEMMs."""
        token_count, top_k = topk_indices.shape
        flat_experts = topk_indices.reshape(-1)
        order = torch.argsort(flat_experts, stable=True)
        token_of = torch.div(order, top_k, rounding_mode="floor")
        counts = torch.bincount(
            flat_experts,
            minlength=self._grouped_w13.shape[0],
        )
        offsets = counts.cumsum(0).to(torch.int32)

        routed_input = latent.index_select(0, token_of)
        gate_value = torch._grouped_mm(
            routed_input,
            self._grouped_w13.transpose(-2, -1),
            offs=offsets,
        ).float()
        gate, value = gate_value.chunk(2, dim=-1)
        routed_hidden = F.silu(gate) * value
        routed_output = torch._grouped_mm(
            routed_hidden.to(latent.dtype),
            self._grouped_w2.transpose(-2, -1),
            offs=offsets,
        )
        routed_output = self.fc2_latent_proj(routed_output)
        routed_output = routed_output * (
            topk_weights.reshape(-1)
            .index_select(0, order)
            .to(routed_output.dtype)
            .unsqueeze(-1)
        )
        output = torch.zeros(
            token_count,
            self.fc2_latent_proj.out_features,
            device=latent.device,
            dtype=acc_dtype,
        )
        start = 0
        for count in counts.tolist():
            if count:
                end = start + count
                token_indices = token_of.narrow(0, start, count)
                output[token_indices] += routed_output[start:end].to(acc_dtype)
                start = end
        return output

    def forward(
        self,
        hidden: torch.Tensor,
        correction_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, H = hidden.shape
        flat = hidden.view(B * T, H)

        # Shared expert: takes hidden directly
        shared_out = self.shared_experts(flat)  # [B*T, H]

        # Routing: from hidden
        topk_indices, topk_weights = self.gate(flat, correction_bias)  # [B*T, k] each

        # Latent projection for expert computation
        latent = self.fc1_latent_proj(flat)     # [B*T, moe_latent_size]

        # Dispatch to experts.
        # TARGET_MOE_FP32ACC=1 accumulates the 8-way expert combine in fp32 instead of bf16.
        # Each token sums top_k=8 weighted expert outputs; in bf16 every partial sum is rounded
        # to 8 mantissa bits, and the rounding depends on the order experts are visited.  This is
        # one of the remaining cross-framework difference sources (PaddleFleet combines inside a
        # fused/grouped kernel).  Off by default so the reference path stays unchanged.
        acc_dtype = torch.float32 if MOE_FP32_ACC else hidden.dtype
        if self._grouped_w13 is not None:
            expert_out = self._forward_grouped(
                latent, topk_indices, topk_weights, acc_dtype
            )
        else:
            expert_out = torch.zeros(B * T, H, device=hidden.device, dtype=acc_dtype)
            for e_idx in range(self.gate.n_routed_experts):
                mask = (topk_indices == e_idx).any(dim=-1)  # [B*T]
                if not mask.any():
                    continue
                e_latent = latent[mask]
                e_out_latent = self.experts[e_idx](e_latent)       # [n, moe_latent]
                e_out = self.fc2_latent_proj(e_out_latent)          # [n, hidden]
                # Get per-token weights for this expert
                expert_mask_2d = (topk_indices == e_idx)           # [B*T, k]
                w = topk_weights[mask][expert_mask_2d[mask]].unsqueeze(-1)
                expert_out[mask] += (e_out * w).to(acc_dtype)

        return (expert_out + shared_out.to(acc_dtype)).to(hidden.dtype).view(B, T, H)


# ============================================================================
# MLA (Multi-head Latent Attention) with gated attention
# ============================================================================

class TargetAttention(nn.Module):
    """MLA attention:
      Q: hidden → q_a_proj [lora_rank] → q_b_proj [n_heads * head_dim]  + RoPE on rope slice
      KV: hidden → kv_a_proj [kv_lora_rank + rope_head_dim] → kv_b_proj [n_heads * (nope+v)]
      gated attn: gate_proj on v, element-wise sigmoid gate
    """

    def __init__(self, config: TargetConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.v_head_dim = config.v_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.use_gated_attn = config.use_gated_attn
        self.use_qk_norm = config.use_qk_norm
        self.rope_interleave = config.rope_interleave

        # Q path
        self.q_a_proj = nn.Linear(config.hidden_size, config.q_lora_rank, bias=False)
        self.q_a_layernorm = TargetRMSNorm(config.q_lora_rank, config.rms_norm_eps)
        # q_b_proj output: n_heads * (qk_nope_head_dim + qk_rope_head_dim) = 16*(192+64) = 4096
        self.q_b_proj = nn.Linear(
            config.q_lora_rank,
            config.num_attention_heads * (config.qk_nope_head_dim + config.qk_rope_head_dim),
            bias=False,
        )

        # KV path
        self.kv_a_proj_with_mqa = nn.Linear(
            config.hidden_size,
            config.kv_lora_rank + config.qk_rope_head_dim,
            bias=False,
        )
        self.kv_a_layernorm = TargetRMSNorm(config.kv_lora_rank, config.rms_norm_eps)
        self.kv_b_proj = nn.Linear(
            config.kv_lora_rank,
            config.num_attention_heads * (config.qk_nope_head_dim + config.v_head_dim),
            bias=False,
        )

        # Output projection
        self.o_proj = nn.Linear(
            config.num_attention_heads * config.v_head_dim,
            config.hidden_size,
            bias=False,
        )

        # Gated attention gate projection
        # Confirmed from ckpt: gate_proj [4096, 1024] = (n_heads*v_head_dim, hidden)
        # Input to gate is hidden_states, output gates the v result
        if config.use_gated_attn:
            self.gate_proj = nn.Linear(
                config.hidden_size,
                config.num_attention_heads * config.v_head_dim,
                bias=False,
            )

        self.softmax_scale = 1.0 / math.sqrt(config.qk_nope_head_dim + config.qk_rope_head_dim)

    def forward(
        self,
        hidden: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        q_position_ids: Optional[torch.Tensor] = None,
        # Optional separate attention memory. Echo MTP computes Q from the
        # fused draft token but reads K/V from the target Layer 23 cache.
        kv_hidden: Optional[torch.Tensor] = None,
        kv_position_ids: Optional[torch.Tensor] = None,
        frozen_key_value: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        B, T, _ = hidden.shape

        # Q
        q_latent = self.q_a_layernorm(self.q_a_proj(hidden))
        q_full = self.q_b_proj(q_latent)
        q = q_full.view(
            B, T, self.num_heads,
            self.qk_nope_head_dim + self.qk_rope_head_dim,
        )

        # KV can come from a separate target-layer memory in Echo mode.
        kv_source = hidden if kv_hidden is None else kv_hidden
        kv_T = kv_source.shape[1]
        kv_input = self.kv_a_proj_with_mqa(kv_source)
        kv_latent = kv_input[..., :self.kv_lora_rank]
        k_rope = kv_input[..., self.kv_lora_rank:]
        kv_latent = self.kv_a_layernorm(kv_latent)
        kv_full = self.kv_b_proj(kv_latent)
        kv_full = kv_full.view(
            B, kv_T, self.num_heads,
            self.qk_nope_head_dim + self.v_head_dim,
        )
        k_nope = kv_full[..., :self.qk_nope_head_dim]
        v = kv_full[..., self.qk_nope_head_dim:]

        q_nope = q[..., :self.qk_nope_head_dim]
        q_rope = q[..., self.qk_nope_head_dim:]

        q_position_ids = _normalize_position_ids(
            q_position_ids, B, T, hidden.device, "q_position_ids"
        )
        cos_q, sin_q = _select_rotary(cos, sin, q_position_ids, B, T)
        q_rope = _apply_rotary(q_rope, cos_q, sin_q, self.rope_interleave)

        kv_position_ids = _normalize_position_ids(
            kv_position_ids, B, kv_T, hidden.device, "kv_position_ids"
        )
        cos_k, sin_k = _select_rotary(cos, sin, kv_position_ids, B, kv_T)
        k_rope = k_rope.unsqueeze(2).expand(-1, -1, self.num_heads, -1)
        k_rope_expanded = _apply_rotary(
            k_rope, cos_k, sin_k, self.rope_interleave
        )

        q_full2 = torch.cat([q_nope, q_rope], dim=-1)
        k_full = torch.cat([k_nope, k_rope_expanded], dim=-1)

        current_kv = (k_full, v)
        if past_key_value is not None:
            if frozen_key_value:
                k_full, v = past_key_value
            else:
                k_full = torch.cat([past_key_value[0], k_full], dim=1)
                v = torch.cat([past_key_value[1], v], dim=1)
        new_kv = (
            current_kv
            if use_cache and frozen_key_value
            else ((k_full, v) if use_cache else None)
        )

        q_t = q_full2.transpose(1, 2)
        k_t = k_full.transpose(1, 2)
        v_t = v.transpose(1, 2)
        # SDPA only when the mask is pure-causal (is_causal=True keeps us on the fused kernel).
        # Anything else -- incremental decode (q_len=1 != kv_len), key padding, custom masks --
        # falls back to EAGER rather than passing an additive mask to SDPA: letting PyTorch
        # auto-select a backend for that case picked cuDNN and failed with
        # `mha_graph->execute(...) is_good() == false`.
        use_sdpa = ATTN_IMPL != "eager" and _is_pure_causal(
            attention_mask, q_t.shape[2], k_t.shape[2]
        )
        if use_sdpa:
            with _sdpa_ctx():
                attn_out = F.scaled_dot_product_attention(
                    q_t, k_t, v_t, attn_mask=None, is_causal=True,
                    scale=self.softmax_scale,
                )
        else:
            attn_weights = torch.matmul(q_t, k_t.transpose(-2, -1)) * self.softmax_scale
            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask
            attn_weights = F.softmax(
                attn_weights, dim=-1, dtype=torch.float32
            ).to(q_t.dtype)
            attn_out = torch.matmul(attn_weights, v_t)
        attn_out = attn_out.transpose(1, 2).contiguous()
        attn_out = attn_out.view(B, T, self.num_heads * self.v_head_dim)

        if self.use_gated_attn:
            gate = torch.sigmoid(self.gate_proj(hidden))
            attn_out = attn_out * gate

        return self.o_proj(attn_out), new_kv


# ============================================================================
# Decoder layer
# ============================================================================

class TargetDecoderLayer(nn.Module):
    def __init__(self, config: TargetConfig, layer_idx: int):
        super().__init__()
        self.self_attn = TargetAttention(config, layer_idx)
        self.mlp = TargetMoE(config)
        self.input_layernorm = TargetRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = TargetRMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self,
        hidden: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        q_position_ids: Optional[torch.Tensor] = None,
        kv_hidden: Optional[torch.Tensor] = None,
        kv_position_ids: Optional[torch.Tensor] = None,
        frozen_key_value: bool = False,
        moe_correction_bias: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple]]:
        residual = hidden
        hidden_normed = self.input_layernorm(hidden)
        kv_hidden_normed = (
            self.input_layernorm(kv_hidden) if kv_hidden is not None else None
        )
        attn_out, new_kv = self.self_attn(
            hidden_normed, cos, sin, attention_mask, past_key_value, use_cache,
            q_position_ids=q_position_ids, kv_hidden=kv_hidden_normed,
            kv_position_ids=kv_position_ids, frozen_key_value=frozen_key_value,
        )
        hidden = residual + attn_out

        residual = hidden
        hidden = residual + self.mlp(
            self.post_attention_layernorm(hidden),
            correction_bias=moe_correction_bias,
        )
        return hidden, new_kv


def _alias_module_parameters(destination: nn.Module, source: nn.Module) -> None:
    """Alias matching Parameters while preserving destination buffers."""
    source_params = dict(source.named_parameters())
    for name, destination_param in list(destination.named_parameters()):
        source_param = source_params.get(name)
        if source_param is None or source_param.shape != destination_param.shape:
            raise ValueError(f"Cannot alias Echo parameter {name}")
        owner = destination
        parts = name.split(".")
        for part in parts[:-1]:
            owner = getattr(owner, part)
        owner._parameters[parts[-1]] = source_param


# ============================================================================
# MTP head
# ============================================================================

class TargetMTPHead(nn.Module):
    """Multi-Token Prediction head.

    Fuses previous hidden state (h_{t-1}) and token embedding (e(x_t)):
        fused = FusionLayer(h_{t-1}, e(x_t))
        h_t   = DecoderLayer_L(fused)  ← shared final backbone layer (ECHO)
                                       ← independent layer (DSpark/baseline)

    Stored keys in checkpoint (layer 24):
        enorm.weight               RMSNorm on embedding residual
        hnorm.weight               RMSNorm on hidden residual
        eh_proj.weight [H, 2H]     linear projection fusing h+e → h
        shared_head.norm.weight    final norm before lm_head
        transformer_layer.*        backbone weights (only present when mtp_reuse_last_layer=False)
    """

    def __init__(self, config: TargetConfig, shared_layer: Optional[TargetDecoderLayer] = None):
        super().__init__()
        self.enorm = TargetRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.hnorm = TargetRMSNorm(config.hidden_size, config.rms_norm_eps)
        # eh_proj consumes normalized token embedding first, then hidden state.
        # This matches PaddleFleet training and SGLang DeepseekModelNextN.
        # stored shape in ckpt: [hidden_size, 2*hidden_size] (HF [out,in])
        self.eh_proj = nn.Linear(2 * config.hidden_size, config.hidden_size, bias=False)
        self.shared_head = nn.ModuleDict({"norm": TargetRMSNorm(config.hidden_size, config.rms_norm_eps)})

        if config.mtp_reuse_last_layer:
            # Training aliases Parameters only. Keep a distinct draft module so
            # non-Parameter buffers, notably the router correction bias, remain
            # Layer-24-specific while all learned Parameters share Layer 23.
            assert shared_layer is not None, "shared_layer required for mtp_reuse_last_layer=True"
            self.transformer_layer = TargetDecoderLayer(
                config, layer_idx=config.num_hidden_layers
            )
            _alias_module_parameters(self.transformer_layer, shared_layer)
        else:
            # DSpark/DeepSeek baseline: independent MTP layer
            self.transformer_layer = TargetDecoderLayer(config, layer_idx=config.num_hidden_layers)

    def forward(
        self,
        prev_hidden: torch.Tensor,
        token_emb: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple] = None,
        use_cache: bool = False,
        q_position_ids: Optional[torch.Tensor] = None,
        kv_hidden: Optional[torch.Tensor] = None,
        kv_position_ids: Optional[torch.Tensor] = None,
        frozen_key_value: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple]]:
        # Fuse current token embedding and previous hidden state.
        h_n = self.hnorm(prev_hidden)
        e_n = self.enorm(token_emb)
        fused = self.eh_proj(torch.cat([e_n, h_n], dim=-1))

        out, new_kv = self.transformer_layer(
            fused, cos, sin, attention_mask, past_key_value, use_cache,
            q_position_ids=q_position_ids, kv_hidden=kv_hidden,
            kv_position_ids=kv_position_ids, frozen_key_value=frozen_key_value,
        )
        out = self.shared_head["norm"](out)
        return out, new_kv


# ============================================================================
# Backbone model
# ============================================================================

class TargetModel(nn.Module):
    def __init__(self, config: TargetConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            TargetDecoderLayer(config, i) for i in range(config.num_hidden_layers)
        ])
        self.norm = TargetRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.rotary_emb = TargetRotaryEmbedding(
            dim=config.qk_rope_head_dim,
            max_seq_len=config.max_position_embeddings,
            base=config.rope_theta,
        )

        # MTP heads
        if config.use_mtp and config.num_nextn_predict_layers > 0:
            # For ECHO (mtp_reuse_last_layer=True): share the final backbone layer
            shared_layer = self.layers[-1] if config.mtp_reuse_last_layer else None
            self.mtp_heads = nn.ModuleList([
                TargetMTPHead(config, shared_layer)
                for _ in range(config.num_nextn_predict_layers)
            ])
        else:
            self.mtp_heads = None

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple]] = None,
        use_cache: bool = False,
        return_mtp_hidden: bool = False,
        position_ids: Optional[torch.Tensor] = None,
    ):
        B, T = input_ids.shape
        device = input_ids.device

        if isinstance(past_key_values, Cache):
            legacy_cache = past_key_values.to_legacy_cache()
            past_key_values = list(legacy_cache) if legacy_cache else None
        past_length = past_key_values[0][0].shape[1] if past_key_values else 0
        # `position_ids` is an override for probing absolute-position effects: the same tokens
        # can be placed at arbitrary positions while everything else is held fixed. RoPE is
        # relative, so a constant shift must leave the output unchanged.
        if position_ids is None:
            if attention_mask is not None and attention_mask.ndim == 2:
                if past_length:
                    past_mask = attention_mask[:, :-T]
                    current_mask = attention_mask[:, -T:]
                    position_ids = (
                        past_mask.sum(dim=-1, keepdim=True)
                        + current_mask.cumsum(dim=-1)
                        - 1
                    )
                else:
                    position_ids = attention_mask.cumsum(dim=-1) - 1
                position_ids = position_ids.clamp_min(0)
            else:
                position_ids = torch.arange(
                    past_length, past_length + T, device=device, dtype=torch.long
                )
        else:
            position_ids = _normalize_position_ids(
                position_ids, B, T, device, "position_ids"
            )

        hidden = self.embed_tokens(input_ids)
        cos, sin = self.rotary_emb(int(position_ids.max().item()) + 1, device)

        # Always build the causal component locally. A 2D HF attention mask is
        # converted to an additive key-padding mask before it is combined.
        seq_len = past_length + T
        causal_mask = torch.full(
            (1, 1, T, seq_len), float("-inf"), device=device, dtype=hidden.dtype
        )
        causal_mask = torch.triu(causal_mask, diagonal=1 + past_length)
        if attention_mask is not None:
            if attention_mask.ndim == 2:
                if attention_mask.shape != (B, seq_len):
                    raise ValueError(
                        f"Expected attention_mask shape {(B, seq_len)}, got "
                        f"{tuple(attention_mask.shape)}"
                    )
                padding_mask = attention_mask[:, None, None, :].to(device=device)
                padding_mask = torch.zeros_like(padding_mask, dtype=hidden.dtype).masked_fill(
                    padding_mask == 0, float("-inf")
                )
                causal_mask = causal_mask + padding_mask
                invalid_queries = attention_mask[:, -T:] == 0
                if invalid_queries.any():
                    causal_mask = torch.where(
                        invalid_queries[:, None, :, None],
                        torch.zeros_like(causal_mask),
                        causal_mask,
                    )
            elif attention_mask.ndim == 4:
                causal_mask = causal_mask + attention_mask.to(
                    device=device, dtype=hidden.dtype
                )
            else:
                raise ValueError(
                    "attention_mask must be a 2D key-padding mask or 4D additive mask"
                )

        new_kvs = []
        for i, layer in enumerate(self.layers):
            pkv = past_key_values[i] if past_key_values else None
            # Capture the input to each layer. Echo MTP reuses Layer 23's
            # attention memory, whose KV was projected from this tensor.
            if return_mtp_hidden and i == len(self.layers) - 1:
                last_layer_input = hidden
                self._last_layer_input = hidden
            hidden, nkv = layer(
                hidden, cos, sin, causal_mask, pkv, use_cache,
                q_position_ids=position_ids, kv_position_ids=position_ids,
            )
            if use_cache:
                new_kvs.append(nkv)

        # Store backbone output for MTP.
        # Must be the POST-final-norm hidden: the native Fleet forward feeds its MTP layer
        # exactly the tensor the lm-head consumes (verified identical, 0.000000).  Capturing
        # it pre-norm made bb's L2 5-69x too large (300-3000 vs ~40, cosine 0.74-0.82) and
        # corrupted every MTP draft at depth 0.
        hidden = self.norm(hidden)
        backbone_hidden = hidden

        if return_mtp_hidden and self.mtp_heads is not None:
            return hidden, backbone_hidden, new_kvs if use_cache else None

        return hidden, None, new_kvs if use_cache else None

    def prepare_inference_weights(self) -> None:
        """Pack backbone and MTP routed experts after the checkpoint is loaded."""
        for layer in self.layers:
            layer.mlp.prepare_inference_weights()
        if self.mtp_heads is not None:
            for mtp_head in self.mtp_heads:
                # The decoder Parameters may alias the final backbone layer, but
                # the packed tensors are inference-owned copies. The MTP head's
                # correction-bias buffer remains independent per depth.
                mtp_head.transformer_layer.mlp.prepare_inference_weights()


# ============================================================================
# ForCausalLM
# ============================================================================

class TargetForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = TargetConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = False

    @classmethod
    def _supports_default_dynamic_cache(cls) -> bool:
        # This implementation stores legacy per-layer (K, V) tuples.
        return False

    def __init__(self, config: TargetConfig):
        super().__init__(config)
        self.model = TargetModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def get_output_embeddings(self):
        return self.lm_head

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        # Extra: return MTP draft logits
        return_mtp_logits: bool = False,
    ):
        use_cache = use_cache if use_cache is not None else self.config.use_cache if hasattr(self.config, "use_cache") else True

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("Specify only one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds_used = None
            # standard path
            hidden, backbone_hidden, new_kvs = self.model(
                input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                return_mtp_hidden=return_mtp_logits,
            )
        else:
            raise NotImplementedError("inputs_embeds path not implemented yet")

        logits = self.lm_head(hidden).float()

        # MTP logits for draft (only for speculative decoding, not training)
        mtp_logits = None
        if return_mtp_logits and self.model.mtp_heads is not None and backbone_hidden is not None:
            mtp_logits_list = []
            device = hidden.device
            T = hidden.shape[1]
            cos, sin = self.model.rotary_emb(T, device)
            mtp_mask = torch.triu(
                torch.full((1, 1, T, T), float("-inf"), device=device, dtype=hidden.dtype),
                diagonal=1,
            )
            prev_h = backbone_hidden
            shifted_ids = input_ids.clone()
            if T > 1:
                shifted_ids[:, :-1] = input_ids[:, 1:]
            for mtp_head in self.model.mtp_heads:
                token_emb = self.model.embed_tokens(shifted_ids)
                mtp_h, _ = mtp_head(
                    prev_h, token_emb, cos, sin,
                    attention_mask=mtp_mask, use_cache=False,
                )
                mtp_logits_list.append(self.lm_head(mtp_h).float())
                prev_h = mtp_h
            mtp_logits = mtp_logits_list

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        if return_dict is False:
            return (loss, logits) + ((mtp_logits,) if mtp_logits else ())

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=new_kvs,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        next_sequence_length: Optional[int] = None,
        past_key_values: Optional[List[Tuple]] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        is_first_iteration: Optional[bool] = False,
        **kwargs,
    ):
        if inputs_embeds is not None:
            raise NotImplementedError("inputs_embeds generation is not implemented")
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
        elif next_sequence_length is not None:
            input_ids = input_ids[:, -next_sequence_length:]
        position_ids = kwargs.get("position_ids")
        if position_ids is not None:
            if past_key_values is not None:
                position_ids = position_ids[:, -1:]
            elif next_sequence_length is not None:
                position_ids = position_ids[:, -next_sequence_length:]
        return {
            "input_ids": input_ids.contiguous(),
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "position_ids": position_ids,
            "use_cache": kwargs.get("use_cache", True),
        }

    @torch.no_grad()
    def generate_with_speculative(
        self,
        input_ids: torch.LongTensor,
        max_new_tokens: int = 32,
        draft_depth: int = 1,
        temperature: float = 0.0,
    ) -> Tuple[torch.LongTensor, dict]:
        """Simple greedy speculative decoding using ECHO MTP head.

        Returns (output_ids, stats) where stats contains accept_length info.
        """
        device = input_ids.device
        generated = input_ids.clone()
        stats = {"accepted": 0, "drafted": 0, "steps": 0}

        while generated.shape[1] - input_ids.shape[1] < max_new_tokens:
            # --- Draft phase ---
            draft_tokens = []
            draft_input = generated

            hidden, backbone_h, _ = self.model(draft_input, use_cache=False, return_mtp_hidden=True)
            target_logits = self.lm_head(hidden).float()

            # Greedy target token from last position
            t0_logits = target_logits[:, -1, :]
            t0 = t0_logits.argmax(dim=-1, keepdim=True)  # [B,1]

            if draft_depth == 1 or self.model.mtp_heads is None:
                # No speculative draft, just use target
                generated = torch.cat([generated, t0], dim=1)
                stats["steps"] += 1
                if t0.item() == self.config.eos_token_id:
                    break
                continue

            # Draft depth > 1: use MTP heads
            draft_tokens.append(t0)
            cos, sin = self.model.rotary_emb(draft_input.shape[1], device)
            prev_h = backbone_h
            draft_logits_all = [target_logits]

            for _ in range(draft_depth - 1):
                token_emb = self.model.embed_tokens(draft_tokens[-1])
                # NOTE: head 0 at every depth is a shortcut of this self-test loop,
                # NOT the trained semantics. Training gives each depth its own head
                # (aliased Parameters, but a depth-specific e_score_correction_bias),
                # and `return_mtp_logits` above iterates `for mtp_head in mtp_heads`
                # accordingly. Serving must follow that loop, not this one -- reading
                # this line as authoritative once led to pinning sglang's draft to
                # depth 0's bias.
                mtp_h, _ = self.model.mtp_heads[0](prev_h, token_emb, cos, sin, use_cache=False)
                d_logits = self.lm_head(mtp_h).float()
                d_tok = d_logits[:, -1:, :].argmax(dim=-1)
                draft_tokens.append(d_tok)
                draft_logits_all.append(d_logits)
                prev_h = mtp_h

            # --- Verify phase (greedy: accept if draft == target argmax) ---
            # Run target on drafted tokens
            verify_input = torch.cat([generated] + draft_tokens, dim=1)
            v_hidden, _, _ = self.model(verify_input, use_cache=False, return_mtp_hidden=False)
            v_logits = self.lm_head(v_hidden).float()

            # Check each draft token
            n_accepted = 0
            start = generated.shape[1]
            for i, d_tok in enumerate(draft_tokens):
                target_tok = v_logits[:, start + i - 1, :].argmax(dim=-1, keepdim=True)
                if target_tok.item() == d_tok.item():
                    n_accepted += 1
                    generated = torch.cat([generated, d_tok], dim=1)
                    stats["accepted"] += 1
                    stats["drafted"] += 1
                    if d_tok.item() == self.config.eos_token_id:
                        return generated, stats
                else:
                    # Accept target's correction
                    generated = torch.cat([generated, target_tok], dim=1)
                    stats["drafted"] += len(draft_tokens)
                    break
            else:
                # All accepted, append bonus token
                bonus = v_logits[:, start + len(draft_tokens) - 1, :].argmax(dim=-1, keepdim=True)
                generated = torch.cat([generated, bonus], dim=1)
                stats["drafted"] += len(draft_tokens)

            stats["steps"] += 1
            if generated[:, -1].item() == self.config.eos_token_id:
                break

        if stats["drafted"] > 0:
            stats["accept_rate"] = stats["accepted"] / stats["drafted"]
        else:
            stats["accept_rate"] = 0.0
        return generated, stats
