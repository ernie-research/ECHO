# Copyright 2025 SGLang Team / Baidu the target
# Licensed under the Apache License, Version 2.0
# ==============================================================================
"""Inference-only the target V2 (echo_target) model for SGLang.

Architecture is identical to DeepSeek V3 except for these checkpoint differences:
  1. block_sparse_moe.*  → mlp.*  (namespace rename)
  2. Expert weights: w1/w2/w3  instead of gate_proj/up_proj/down_proj
  3. routed_scaling_factor_param: learnable scalar in ckpt (skip; use config float)
  4. self_attn.gate_proj: gated-attention weight. the target applies
     attn_out *= sigmoid(gate_proj(hidden)) before o_proj. DeepSeek MLA has no
     such gate, so we install it (see _install_gated_attn) and LOAD the weight.
  5. transformer_layer.mlp.gate.routed_scaling_factor_param in MTP layer (skip)

Latent MoE:
  the target uses fc1_latent_proj [hidden=1024→latent=512] + FusedMoE(hidden=512) +
  fc2_latent_proj [latent=512→hidden=1024]. DeepseekV2MoE creates FusedMoE with
  hidden_size=config.hidden_size=1024, so we need TargetMoE to fix this.

ECHO MTP (num_nextn_predict_layers=1, mtp_reuse_last_layer=True):
  Layer 24 stores a full copy of the transformer weights plus MTP-specific keys
  (enorm, hnorm, eh_proj, shared_head.norm).  All weights are directly at
  model.layers.24.* (no sub-prefix), which the DeepseekV2WeightLoaderMixin's
  nextn machinery routes to model.decoder.* / model.{spec_names}.* correctly.
"""
from __future__ import annotations

import logging
import types
from typing import Iterable, Optional, Tuple

import torch
from torch import nn
from transformers import PretrainedConfig

from sglang.srt.distributed import get_pp_group
from sglang.srt.layers.linear import ColumnParallelLinear, ReplicatedLinear
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.mem_cache.memory_pool import KVCache
from sglang.srt.models.deepseek_nextn import DeepseekModelNextN, DeepseekV3ForCausalLMNextN
from sglang.srt.models.deepseek_v2 import (
    DeepseekV2DecoderLayer,
    DeepseekV2MoE,
    DeepseekV2Model,
    DeepseekV2ForCausalLM,
)
from sglang.srt.models.utils import WeightsMapper
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils.common import add_prefix, make_layers

logger = logging.getLogger(__name__)

_SKIP_SUFFIXES = ("routed_scaling_factor_param",)

# the target expert weight naming → SGLang DeepseekV2MoE naming
_EXPERT_W_MAP = {".w1.": ".gate_proj.", ".w2.": ".down_proj.", ".w3.": ".up_proj."}


def _remap_target_key(name: str) -> Optional[str]:
    """Remap one the target V2 checkpoint key.

    Returns the remapped name, or None if the key should be skipped.
    """
    # 1. Skip learnable scaling param (SGLang uses config float; not a parameter)
    for sfx in _SKIP_SUFFIXES:
        if sfx in name:
            return None

    # 2. gate_proj under self_attn: the target checkpoint stores a gated-attention
    #    weight, but the training-source PaddleFleet inference model does NOT
    #    apply it (verified: native oracle has no attn-gate param, and the
    #    PyTorch reference matches it only with the gate DISABLED). Applying it
    #    corrupts the output. Skip loading this key entirely.
    if name.endswith(".self_attn.gate_proj.weight"):
        return None

    # 3. block_sparse_moe → mlp namespace
    name = name.replace("block_sparse_moe.", "mlp.")

    # 4. e_score_correction_bias: checkpoint stores it at mlp.*,
    #    SGLang's MoEGate stores it under mlp.gate.*
    if name.endswith(".mlp.e_score_correction_bias"):
        name = name[: -len("mlp.e_score_correction_bias")] + "mlp.gate.e_score_correction_bias"

    # 5. Expert weight naming: w1→gate_proj, w2→down_proj, w3→up_proj
    #    Applies to both routed experts (mlp.experts.E.wN.) and
    #    shared experts (mlp.shared_experts.wN.)
    for w_src, w_dst in _EXPERT_W_MAP.items():
        if w_src in name:
            name = name.replace(w_src, w_dst)
            break

    return name


def _remap_target_weights(
    weights: Iterable[Tuple[str, torch.Tensor]],
) -> Iterable[Tuple[str, torch.Tensor]]:
    for name, w in weights:
        mapped = _remap_target_key(name)
        if mapped is not None:
            yield mapped, w


def _patch_target_config(config: PretrainedConfig) -> None:
    """Add missing fields that SGLang's DeepSeek model code requires."""
    # the target routing (verified against the training-source oracle):
    #   weights = normalize(raw_sigmoid_scores_of_topk_experts)
    # i.e. the per-layer learnable routed_scaling_factor is applied *before*
    # renormalisation and therefore CANCELS OUT, leaving pure normalised top-k
    # weights. DeepSeek's default applies routed_scaling_factor as a real
    # post-normalisation multiplier (2.5x) and, with norm_topk_prob unset,
    # skips normalisation entirely — both corrupt the MoE output. Force the
    # the target-correct behaviour: normalise, no extra scaling.
    config.norm_topk_prob = True
    config.routed_scaling_factor = 1.0
    if not hasattr(config, "rope_scaling"):
        config.rope_scaling = None
    if not hasattr(config, "num_nextn_predict_layers"):
        config.num_nextn_predict_layers = 1
    # token IDs required by DeepseekV2Model
    if not hasattr(config, "pad_token_id") or config.pad_token_id is None:
        config.pad_token_id = 0
    if not hasattr(config, "bos_token_id") or config.bos_token_id is None:
        config.bos_token_id = 1
    if not hasattr(config, "eos_token_id") or config.eos_token_id is None:
        config.eos_token_id = 2
    # Transformers >=4.51 maps rope_scaling → rope_parameters as a property.
    # DeepseekV2DecoderLayer checks `hasattr(config, "rope_parameters")` and
    # then tries config.rope_parameters["rope_theta"]. We need to ensure
    # rope_parameters is absent (or None stored via __dict__ directly)
    # so the `hasattr` check returns False and the code falls through to
    # rope_theta / rope_scaling path.
    # rope_scaling setter writes to rope_parameters; set it to None via __dict__
    # so hasattr returns True but the value is None — but deepseek code checks
    # `if hasattr(config, "rope_parameters"):` and then does ["rope_theta"].
    # The only safe option: forcibly remove rope_parameters from __dict__.
    config.__dict__.pop("rope_parameters", None)
    # Also ensure rope_theta and rope_scaling are present for the else-branch
    if not hasattr(config, "rope_theta"):
        config.__dict__["rope_theta"] = 10000.0
    config.__dict__["rope_scaling"] = None


class TargetMoE(DeepseekV2MoE):
    """the target V2 Latent MoE.

    the target uses a 2-stage bottleneck:
      fc1_latent_proj: [hidden=1024] → [latent=512]
      FusedMoE experts:  input hidden_size=512, intermediate=1024
      fc2_latent_proj: [latent=512] → [hidden=1024]

    The MoE gate operates on the FULL hidden dimension (1024); only the
    FusedMoE expert kernel uses the latent size (512).

    Implementation: call super().__init__ with the real hidden_size so the gate
    gets shape [n_experts, hidden=1024].  Then swap hidden_size → moe_latent_size
    and replace self.experts with a correctly-sized FusedMoE.
    """

    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        **kwargs,
    ):
        moe_latent_size = getattr(config, "moe_latent_size", None)

        # Step 1: Call super().__init__ with the REAL hidden_size.
        # This creates self.gate with weight [n_experts, hidden=1024] (correct)
        # and self.experts with hidden_size=1024 (will be replaced below).
        super().__init__(
            config=config,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=prefix,
            **kwargs,
        )

        if moe_latent_size is not None:
            orig_hidden_size = config.hidden_size

            # Step 2: Replace self.experts with FusedMoE(hidden_size=moe_latent_size).
            # Temporarily swap config.hidden_size so get_moe_impl_class uses latent size.
            config.hidden_size = moe_latent_size
            from sglang.srt.layers.moe.ep_moe.layer import get_moe_impl_class
            from sglang.srt.server_args import get_global_server_args as _gsa
            from sglang.srt.runtime_context import get_parallel as _gp
            from sglang.srt.layers.moe.utils import RoutingMethodType

            _par = _gp()
            _is_deepep_fusion = False  # the target does not use DeepEP

            num_experts_for_moe = (
                config.n_routed_experts + self.num_fused_shared_experts
            )
            top_k_for_moe = (
                config.num_experts_per_tok + self.num_fused_shared_experts
            )

            self.experts = get_moe_impl_class(quant_config)(
                num_experts=num_experts_for_moe + _gsa().ep_num_redundant_experts,
                num_fused_shared_experts=self.num_fused_shared_experts,
                top_k=top_k_for_moe,
                hidden_size=moe_latent_size,          # <- latent size here
                intermediate_size=config.moe_intermediate_size,
                layer_id=layer_id,
                quant_config=quant_config,
                routed_scaling_factor=self.routed_scaling_factor,
                routing_method_type=getattr(
                    config, "routing_method_type", RoutingMethodType.DeepSeekV3
                ),
                swiglu_limit=getattr(config, "swiglu_limit", None),
                prefix=add_prefix("experts", prefix),
            )

            # Restore real hidden_size
            config.hidden_size = orig_hidden_size

            # Step 3: Add latent projection layers (always replicated, no TP split)
            self.fc1_latent_proj = ReplicatedLinear(
                input_size=orig_hidden_size,
                output_size=moe_latent_size,
                bias=False,
                quant_config=quant_config,
                prefix=add_prefix("fc1_latent_proj", prefix),
            )
            self.fc2_latent_proj = ReplicatedLinear(
                input_size=moe_latent_size,
                output_size=orig_hidden_size,
                bias=False,
                quant_config=quant_config,
                prefix=add_prefix("fc2_latent_proj", prefix),
            )
        else:
            self.fc1_latent_proj = None
            self.fc2_latent_proj = None

    # -------------------------------------------------------------------------
    # Override forward helpers to inject latent projections around experts
    # -------------------------------------------------------------------------

    def _experts_forward(self, hidden_states, topk_output):
        """Run fc1_latent → experts → fc2_latent around the FusedMoE call."""
        if self.fc1_latent_proj is not None:
            hidden_states, _ = self.fc1_latent_proj(hidden_states)
        final = self.experts(hidden_states, topk_output)
        if self.fc2_latent_proj is not None:
            final, _ = self.fc2_latent_proj(final)
        return final

    def forward_normal(self, hidden_states, should_allreduce_fusion=False,
                       use_reduce_scatter=False, gemm_output_zero_allocator=None,
                       input_ids=None, input_ids_global=None):
        # Re-implement forward_normal with latent projection injected.
        # We call DeepseekV2MoE.forward_normal but override the experts call
        # by monkey-patching self.experts temporarily.
        # Use object.__setattr__ to bypass nn.Module's Module-type enforcement.
        _orig_experts = self.experts

        class _LatentExpertsProxy:
            """Wraps FusedMoE to inject latent projection."""
            def __init__(proxy, fused_moe, fc1, fc2):
                proxy._moe = fused_moe
                proxy._fc1 = fc1
                proxy._fc2 = fc2
                # forward attributes that DeepseekV2MoE.forward_normal uses
                proxy.moe_runner_config = fused_moe.moe_runner_config
                proxy.supports_deferred_finalize = getattr(
                    fused_moe, "supports_deferred_finalize", False)
                proxy.dispatcher = getattr(fused_moe, "dispatcher", None)
                proxy.quant_method = getattr(fused_moe, "quant_method", None)
                proxy.should_fuse_routed_scaling_factor_in_topk = getattr(
                    fused_moe, "should_fuse_routed_scaling_factor_in_topk", False)
                proxy.num_local_experts = getattr(fused_moe, "num_local_experts", 0)

            def __call__(proxy, hidden_states, topk_output, **kw):
                if proxy._fc1 is not None:
                    hidden_states, _ = proxy._fc1(hidden_states)
                out = proxy._moe(hidden_states, topk_output, **kw)
                if proxy._fc2 is not None:
                    out, _ = proxy._fc2(out)
                return out

            def named_parameters(proxy, *a, **kw):
                return proxy._moe.named_parameters(*a, **kw)

        proxy = _LatentExpertsProxy(
            _orig_experts, self.fc1_latent_proj, self.fc2_latent_proj)
        object.__setattr__(self, "experts", proxy)
        try:
            result = super().forward_normal(
                hidden_states,
                should_allreduce_fusion=should_allreduce_fusion,
                use_reduce_scatter=use_reduce_scatter,
                gemm_output_zero_allocator=gemm_output_zero_allocator,
                input_ids=input_ids,
                input_ids_global=input_ids_global,
            )
        finally:
            object.__setattr__(self, "experts", _orig_experts)
        return result

    def forward_normal_dual_stream(self, hidden_states, should_allreduce_fusion=False,
                                   use_reduce_scatter=False, gemm_output_zero_allocator=None,
                                   input_ids=None, input_ids_global=None):
        _orig_experts = self.experts

        class _LatentExpertsProxy:
            def __init__(proxy, fused_moe, fc1, fc2):
                proxy._moe = fused_moe
                proxy._fc1 = fc1
                proxy._fc2 = fc2
                proxy.moe_runner_config = fused_moe.moe_runner_config
                proxy.supports_deferred_finalize = getattr(
                    fused_moe, "supports_deferred_finalize", False)
                proxy.dispatcher = getattr(fused_moe, "dispatcher", None)
                proxy.quant_method = getattr(fused_moe, "quant_method", None)
                proxy.should_fuse_routed_scaling_factor_in_topk = getattr(
                    fused_moe, "should_fuse_routed_scaling_factor_in_topk", False)
                proxy.num_local_experts = getattr(fused_moe, "num_local_experts", 0)

            def __call__(proxy, hidden_states, topk_output, **kw):
                if proxy._fc1 is not None:
                    hidden_states, _ = proxy._fc1(hidden_states)
                out = proxy._moe(hidden_states, topk_output, **kw)
                if proxy._fc2 is not None:
                    out, _ = proxy._fc2(out)
                return out

            def named_parameters(proxy, *a, **kw):
                return proxy._moe.named_parameters(*a, **kw)

        proxy = _LatentExpertsProxy(
            _orig_experts, self.fc1_latent_proj, self.fc2_latent_proj)
        object.__setattr__(self, "experts", proxy)
        try:
            result = super().forward_normal_dual_stream(
                hidden_states,
                should_allreduce_fusion=should_allreduce_fusion,
                use_reduce_scatter=use_reduce_scatter,
                gemm_output_zero_allocator=gemm_output_zero_allocator,
                input_ids=input_ids,
                input_ids_global=input_ids_global,
            )
        finally:
            object.__setattr__(self, "experts", _orig_experts)
        return result


def _install_gated_attn(attn, config, quant_config, prefix):
    """Install the target gated attention onto a DeepseekV2AttentionMLA instance.

    the target computes ``attn_out *= sigmoid(gate_proj(hidden))`` element-wise on the
    ``[tokens, num_heads * v_head_dim]`` attention output BEFORE ``o_proj``.
    DeepSeek MLA lacks this, so we:
      1. create a ColumnParallelLinear ``gate_proj`` (same TP sharding as o_proj's
         input), so the ckpt key ``...self_attn.gate_proj.weight`` loads normally;
      2. wrap ``forward_prepare`` to compute the per-token gate from the attention
         input hidden states and stash it on the module;
      3. wrap ``o_proj.forward`` to multiply its input by the stashed gate.
    This covers every MLA code path (MHA prefill / MLA-absorb decode) uniformly
    because they all funnel through ``self.o_proj(attn_output)``.
    """
    from sglang.srt.runtime_context import get_parallel as _get_parallel

    par = _get_parallel()
    gate_out = config.num_attention_heads * config.v_head_dim
    attn.attn_out_gate = ColumnParallelLinear(
        config.hidden_size,
        gate_out,
        bias=False,
        quant_config=quant_config,
        prefix=add_prefix("attn_out_gate", prefix),
        tp_rank=par.attn_tp_rank,
        tp_size=par.attn_tp_size,
    )
    object.__setattr__(attn, "_target_gate", None)

    orig_prepare = attn.forward_prepare

    def gated_prepare(self, positions, hidden_states, forward_batch, zero_allocator,
                      *args, **kwargs):
        if isinstance(hidden_states, torch.Tensor) and hidden_states.shape[0] > 0:
            g, _ = self.attn_out_gate(hidden_states)
            object.__setattr__(self, "_target_gate", torch.sigmoid(g))
        else:
            object.__setattr__(self, "_target_gate", None)
        return orig_prepare(positions, hidden_states, forward_batch,
                            zero_allocator, *args, **kwargs)

    attn.forward_prepare = types.MethodType(gated_prepare, attn)

    o_proj = attn.o_proj
    orig_o_forward = o_proj.forward
    import os as _os
    _dbg = _os.environ.get("TARGET_GATE_DEBUG") == "1"
    _tag = prefix

    def gated_o_forward(x):
        g = attn._target_gate
        if _dbg and getattr(attn, "_dbg_n", 0) < 2:
            try:
                w = attn.attn_out_gate.weight
                gm = g.float().mean().item() if g is not None else -1
                print(f"[GATE {_tag}] x.shape={tuple(x.shape)} "
                      f"gate_mean={gm:.4f} gate_none={g is None} "
                      f"w.norm={w.float().norm().item():.3f} w.shape={tuple(w.shape)}",
                      flush=True)
                object.__setattr__(attn, "_dbg_n", getattr(attn, "_dbg_n", 0) + 1)
            except Exception as _e:
                print(f"[GATE {_tag}] dbg err {_e}", flush=True)
        if g is not None:
            x = x * g
        return orig_o_forward(x)

    o_proj.forward = gated_o_forward


class TargetDecoderLayer(DeepseekV2DecoderLayer):
    """Identical to DeepseekV2DecoderLayer except sparse layers use TargetMoE
    and self-attention gets the target gated attention installed."""

    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        **kwargs,
    ) -> None:
        super().__init__(config=config, layer_id=layer_id, quant_config=quant_config, **kwargs)
        # Replace the MoE instance created by super().__init__ with TargetMoE
        if self.is_layer_sparse and isinstance(self.mlp, DeepseekV2MoE):
            moe_prefix = add_prefix("mlp", kwargs.get("prefix", f"layers.{layer_id}"))
            self.mlp = TargetMoE(
                config=config,
                layer_id=layer_id,
                quant_config=quant_config,
                prefix=moe_prefix,
                alt_stream=getattr(self.mlp, "alt_stream", None),
                is_nextn=self.is_nextn,
            )
        # NOTE: the target gated attention is intentionally NOT installed. The
        # training-source PaddleFleet inference model does not apply the ckpt's
        # self_attn.gate_proj weight; enabling it corrupts the forward output.


class TargetModel(DeepseekV2Model):
    """DeepseekV2Model but uses TargetDecoderLayer for all layers."""

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        # Call nn.Module.__init__ + manually re-do DeepseekV2Model.__init__
        # replacing DeepseekV2DecoderLayer with TargetDecoderLayer.
        # We call super().__init__ which internally calls make_layers with
        # DeepseekV2DecoderLayer. Then we rebuild layers with our subclass.
        super().__init__(config=config, quant_config=quant_config, prefix=prefix)
        # Rebuild the layers with the correct decoder layer class
        # (super already sets self.start_layer / self.end_layer / alt_stream)
        from sglang.srt.layers.utils.cp_utils import is_prefill_context_parallel_enabled
        # is_dsa_enable_prefill_cp not available in this SGLang version; the target is not DSA
        dsa_enable_prefill_cp = False
        mla_enable_prefill_cp = is_prefill_context_parallel_enabled()
        self.layers, self.start_layer, self.end_layer = make_layers(
            config.num_hidden_layers,
            lambda idx, prefix: TargetDecoderLayer(
                config=config,
                layer_id=idx,
                quant_config=quant_config,
                prefix=prefix,
                alt_stream=self.alt_stream,
                dsa_enable_prefill_cp=dsa_enable_prefill_cp,
                mla_enable_prefill_cp=mla_enable_prefill_cp,
            ),
            pp_rank=self.pp_group.rank_in_group,
            pp_size=self.pp_group.world_size,
            prefix=add_prefix("layers", prefix),
        )


class TargetForCausalLM(DeepseekV2ForCausalLM):
    """the target V2 causal LM.

    Inherits everything from DeepseekV2ForCausalLM; uses TargetModel
    (which swaps in TargetMoE with latent projections) and remaps
    checkpoint keys on load.
    """

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        _patch_target_config(config)
        super().__init__(config, quant_config=quant_config, prefix=prefix)
        # Replace the model created by super() with TargetModel
        self.model = TargetModel(
            config, quant_config, prefix=add_prefix("model", prefix)
        )

    def determine_num_fused_shared_experts(self, architecture="TargetForCausalLM"):
        self.num_fused_shared_experts = 0
        from sglang.srt.server_args import get_global_server_args
        get_global_server_args().disable_shared_experts_fusion = True

    def load_weights(
        self, weights: Iterable[Tuple[str, torch.Tensor]], is_nextn: bool = False
    ):
        remapped = _remap_target_weights(weights)
        self.do_load_weights(remapped, is_nextn=is_nextn)


class TargetModelNextN(DeepseekModelNextN):
    """DeepseekModelNextN but uses TargetDecoderLayer for the decoder layer.

    the target uses Latent MoE (hidden=1024, moe_latent=512) so we must replace the
    plain DeepseekV2DecoderLayer with TargetDecoderLayer which carries the
    fc1/fc2_latent_proj wrappers.
    """

    def __init__(
        self,
        config,
        quant_config=None,
        prefix: str = "",
    ) -> None:
        super().__init__(config=config, quant_config=quant_config, prefix=prefix)
        # Replace the plain decoder with an TargetDecoderLayer
        from sglang.srt.utils import add_prefix
        from sglang.srt.layers.utils.cp_utils import is_prefill_context_parallel_enabled
        from sglang.srt.models.deepseek_nextn import DeepseekModelNextN
        layer_name = "decoder"
        self.decoder = TargetDecoderLayer(
            config=config,
            layer_id=0,
            quant_config=quant_config,
            is_nextn=True,
            prefix=add_prefix(layer_name, prefix),
            alt_stream=self.alt_stream,
            dsa_enable_prefill_cp=False,
            mla_enable_prefill_cp=is_prefill_context_parallel_enabled(),
        )


class TargetForCausalLMNextN(DeepseekV3ForCausalLMNextN):
    """the target V2 ECHO-MTP draft model for SGLang speculative decoding."""

    hf_to_sglang_mapper = WeightsMapper(
        orig_to_new_substr={
            "model.layers.24": "model.decoder",
        },
    )

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        _patch_target_config(config)
        nn.Module.__init__(self)
        self.config = config
        self.tp_size = get_parallel().tp_size
        self.quant_config = quant_config
        self.pp_group = get_pp_group()
        self.num_fused_shared_experts = 0

        from sglang.srt.layers.utils.cp_utils import is_prefill_context_parallel_enabled
        from sglang.srt.layers.logits_processor import LogitsProcessor
        from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead
        from sglang.srt.server_args import get_global_server_args

        self.use_dsa = False
        self.dsa_enable_prefill_cp = False
        self.mla_enable_prefill_cp = is_prefill_context_parallel_enabled()
        if self.mla_enable_prefill_cp:
            self.cp_rank = get_parallel().attn_cp_rank
            self.cp_size = get_parallel().attn_cp_size
        else:
            self.cp_rank = None
            self.cp_size = None

        self.model = TargetModelNextN(
            config, quant_config, prefix=add_prefix("model", prefix)
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=add_prefix("model.shared_head.head", prefix),
            use_attn_tp_group=get_global_server_args().enable_dp_lm_head,
        )
        self.logits_processor = LogitsProcessor(config)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        remapped = list(_remap_target_weights(weights))
        super().load_weights(iter(remapped))

    def determine_num_fused_shared_experts(self, architecture="TargetForCausalLMNextN"):
        self.num_fused_shared_experts = 0

    # ------------------------------------------------------------------
    # FROZEN_KV_MTP interface
    # ------------------------------------------------------------------

    @property
    def backbone_hidden_size(self) -> int:
        """Hidden size of the *target* backbone, used by FrozenKVMTPDraftWorker
        to size the recurrent hidden state buffer."""
        return int(self.config.hidden_size)

    def get_embed_and_head(self):
        return self.model.embed_tokens.weight, self.lm_head.weight

    def set_embed_and_head(self, embed, head):
        """Bind target embed/head weights (called by FrozenKVMTPDraftWorker)."""
        del self.model.embed_tokens.weight
        del self.lm_head.weight
        self.model.embed_tokens.weight = embed
        self.lm_head.weight = head
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    def build_frozen_kv_mtp_context(
        self,
        target_model,
        target_token_to_kv_pool: KVCache,
    ):
        """Map the single draft decoder layer to target layer 23 (index 23, 0-based).

        ECHO-MTP reuses the target backbone's last layer (layer_id=23 in a
        24-layer model) for both target decoding and draft speculation.  The
        frozen-KV worker must therefore point the draft attention to the same
        KV cache slot that the target layer 23 writes into.
        """
        from sglang.srt.speculative.frozen_kv_mtp_info import FrozenKVMTPContext

        last_layer_id = int(self.config.num_hidden_layers) - 1  # 23 for 24-layer model
        # The draft model has exactly 1 decoder layer (logical index 0).
        physical_layer_ids = {0: last_layer_id}
        return FrozenKVMTPContext(
            target_token_to_kv_pool=target_token_to_kv_pool,
            physical_layer_ids=physical_layer_ids,
        )

    def bind_frozen_kv_context(self, ctx) -> None:
        """Wire draft attention to read the target's last-layer KV cache."""
        decoder_layer = self.model.decoder
        attn = getattr(decoder_layer, "self_attn", None)
        if attn is None:
            logger.warning(
                "TargetForCausalLMNextN.bind_frozen_kv_context: "
                "decoder has no self_attn attribute; skipping KV bind."
            )
            return
        target_phys = ctx.get_physical_layer_id(0)
        attn.is_kv_shared_layer = True
        attn.kv_shared_layer_index = target_phys
        attn.layer_id = 0
        # DeepseekV2AttentionMLA uses attn_mqa + attn_mha instead of a single attn.
        for sub_attn_name in ("attn_mqa", "attn_mha", "attn"):
            sub_attn = getattr(attn, sub_attn_name, None)
            if sub_attn is not None:
                sub_attn.layer_id = target_phys


EntryClass = [TargetForCausalLM, TargetForCausalLMNextN]
