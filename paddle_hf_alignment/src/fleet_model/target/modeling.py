"""Target MoE PaddleFleet model classes.
Matches the PaddleFormers pattern (e.g. ``qwen3_moe/modeling.py``):
provider + capabilities + ForCausalLM, all in one module.
"""

import logging
import math
from dataclasses import dataclass
from functools import partial

from paddleformers.transformers.gpt_provider import GPTModelProvider
from paddleformers.transformers.model_utils import PretrainedModel
from paddlefleet.transformer.utils import is_layer_window_attention


logger = logging.getLogger(__name__)


def _get_layer_attn_split_info(config, layer_idx):
    """Compute per-layer attention split info (SWA-aware).

    Returns (num_attention_heads, num_key_value_heads, num_query_groups, split_dims, layer_is_swa).
    ``split_dims`` has length 4 (gated: Q, Gate, K, V) or 3 (Q, K, V).
    ``layer_idx`` is the *real* layer index (already including ``num_empty_layers_add_in_head``).
    """
    layer_is_swa = is_layer_window_attention(
        getattr(config, "sliding_window", None),
        getattr(config, "window_attn_skip_freq", None),
        layer_idx - getattr(config, "num_empty_layers_add_in_head", 0),
    )
    if layer_is_swa:
        head_dim = config.swa_head_dim
        v_head_dim = config.swa_v_head_dim
        num_attention_heads = config.swa_num_attention_heads
        num_key_value_heads = config.swa_num_key_value_heads
    else:
        head_dim = config.head_dim
        v_head_dim = config.v_head_dim
        num_attention_heads = config.num_attention_heads
        num_key_value_heads = config.num_key_value_heads

    gated_attn = getattr(config, "use_gated_attn", False) or getattr(config, "gated_attention", False)
    num_query_groups = num_key_value_heads
    q_heads_per_group = num_attention_heads // num_key_value_heads
    heads_per_group = q_heads_per_group
    if getattr(config, "use_vha_attention", False):
        q_heads_per_group = q_heads_per_group // num_key_value_heads
    q_dim = q_heads_per_group * head_dim

    if gated_attn:
        gate_dim = heads_per_group * v_head_dim
        split_dims = [q_dim, gate_dim, head_dim, v_head_dim]
    else:
        split_dims = [q_dim, head_dim, v_head_dim]

    return num_attention_heads, num_key_value_heads, num_query_groups, split_dims, layer_is_swa


# ===========================================================================
# TargetProvider
# ===========================================================================


@dataclass
class TargetProvider(GPTModelProvider):
    """PaddleFleet provider for the target."""

    model_type = "echo_target"

    transform_rules = {
        **GPTModelProvider.transform_rules,
        "initializer_range": "init_method_std",
        "moe_layer_interval": "moe_layer_freq",
        "attention_probs_dropout_prob": "attention_dropout",
        "use_gated_attn": "gated_attention",
        "moe_norm_gate_logits": "norm_topk_prob",
        "dtype": "params_dtype",
    }

    moe_router_load_balancing_type: str = "seq_aux_loss"
    gated_linear_unit: bool = True
    bias_activation_fusion: bool = True
    rotary_base: float = 1000000.0
    rotary_percent: float = 0.5
    moe_shared_expert_overlap: bool = True
    moe_router_pre_softmax: bool = False
    moe_permute_fusion: bool = True
    moe_router_dtype: str = "fp32"
    moe_router_enable_expert_bias: bool = True
    moe_router_bias_update_rate: float = 0
    persist_layer_norm: bool = True
    moe_router_force_load_balancing: bool = True
    share_embeddings_and_output_weights: bool = False
    apply_rope_fusion: bool = True
    mtp_loss_scaling_factor: float = 0.1
    recompute_granularity: str = None
    virtual_pipeline_model_parallel_size: int = None
    rope_scaling: float = 1.0
    bias_dropout_fusion: bool = True
    moe_expert_fusion: bool = False
    fp32_residual_connection: bool = False


# ===========================================================================
# Provider config helpers
# ===========================================================================


def _maybe_map_legacy_config(provider, config, old_name, new_name):
    explicit_keys = getattr(config, "_explicit_model_config_keys", set())
    if old_name in explicit_keys and new_name in explicit_keys:
        old_value = getattr(config, old_name)
        new_value = getattr(config, new_name)
        if old_value != new_value:
            raise ValueError(
                f"'{old_name}' and '{new_name}' are both set but differ: "
                f"{old_value} != {new_value}. Please keep only '{new_name}'."
            )
    if new_name in explicit_keys:
        return
    if old_name in explicit_keys and hasattr(config, old_name):
        setattr(provider, new_name, getattr(config, old_name))


def apply_target_config_overrides(provider, config):
    """Apply the target fields that need explicit compatibility handling."""
    provider.init_method_std = config.initializer_range

    if hasattr(config, "moe_grouped_gemm"):
        provider.moe_grouped_gemm = config.moe_grouped_gemm

    provider.moe_layer_freq = config.moe_layer_interval
    provider.attention_dropout = config.attention_probs_dropout_prob
    provider.gated_attention = config.use_gated_attn
    provider.norm_topk_prob = config.moe_norm_gate_logits

    _maybe_map_legacy_config(provider, config, "fused_moe_topk", "moe_topk_fusion")
    _maybe_map_legacy_config(provider, config, "fused_routing_map", "routing_map_fusion")
    _maybe_map_legacy_config(provider, config, "fused_qk_norm", "qk_norm_fusion")

    if hasattr(config, "partial_rotary_factor"):
        provider.rotary_percent = config.partial_rotary_factor

    return provider


def apply_training_args_to_provider(provider, training_args):
    """Apply training/runtime arguments to provider."""
    provider.autocast_dtype = "bfloat16"
    provider.params_dtype = "bfloat16" if training_args.bf16 else "float32"
    provider.bf16 = training_args.bf16
    provider.fp16 = training_args.fp16
    provider.pipeline_model_parallel_size = training_args.pipeline_model_parallel_size
    provider.tensor_model_parallel_size = training_args.tensor_model_parallel_size
    provider.expert_model_parallel_size = training_args.expert_model_parallel_size
    provider.virtual_pipeline_model_parallel_size = training_args.virtual_pipeline_model_parallel_size
    provider.sequence_parallel = training_args.sequence_parallel
    provider.max_sequence_length = training_args.max_seq_length
    provider.seq_length = training_args.max_seq_length
    provider.overlap_p2p_comm = training_args.overlap_p2p_comm
    provider.cp_comm_type = training_args.cp_comm_type
    provider.deterministic_mode = training_args.deterministic_mode
    provider.batch_p2p_comm = training_args.batch_p2p_comm
    provider.variable_seq_lengths = training_args.variable_seq_lengths
    provider.microbatch_group_size_per_vp_stage = training_args.pipeline_model_parallel_size
    provider.flashmask_use_varlen = training_args.flashmask_use_varlen
    return provider


def apply_final_precision_overrides(provider, config):
    """Apply final provider overrides that the old hard bridge used."""
    provider.bf16 = False
    provider.rope_scaling = None
    provider.moe_layer_freq = config.moe_layer_freq
    provider.gpt_model_use_experimental_version = getattr(config, "gpt_model_use_experimental_version", False)
    provider.gated_linear_unit = True
    provider.fp32_residual_connection = False
    return provider


def validate_target_provider(provider, config, training_args=None):
    """Validate EB5 provider compatibility with current PaddleFleet support."""
    if config.use_gated_attn and config.gated_attn_act != "sigmoid":
        raise NotImplementedError("Only support gated_attn_act=sigmoid in PaddleFleet right now.")

    if not getattr(config, "mtp_attention_flexible", True):
        raise NotImplementedError("Only support mtp_attention_flexible=True in PaddleFleet right now.")

    if not getattr(config, "moe_dequant_input", True):
        raise NotImplementedError(
            "Fleet backend only supports moe_dequant_input=True. "
            "Remove this field from yaml/model_config.json (Fleet path will default it to True), "
            "or explicitly set it to True."
        )

    if config.use_ep_comm_overlap:
        raise NotImplementedError(
            "Not support use_ep_comm_overlap in PaddleFleet right now. "
            "If you want to open it, set moe_shared_expert_overlap=True in model_args.model_config."
        )

    if config.topk_method == "noaux_tc" and provider.topk_method != "noaux_tc":
        raise NotImplementedError("Only support topk_method=noaux_tc in PaddleFleet right now.")

    if training_args is None:
        return

    dw_p2p_overlap = getattr(config, "dw_p2p_overlap", False)
    if not dw_p2p_overlap:
        return

    pp_degree = training_args.pipeline_model_parallel_size
    accumulate_steps = training_args.gradient_accumulation_steps

    if not training_args.best_unbalanced_scheduler:
        raise NotImplementedError("Only support dw_p2p_overlap=True if best_unbalanced_scheduler=True.")
    if accumulate_steps >= 2 * pp_degree:
        raise NotImplementedError("Only support dw_p2p_overlap=True if acc < 2 * pp_degree.")
    if training_args.tensor_model_parallel_size > 1:
        raise NotImplementedError("Only support dw_p2p_overlap=True if tensor_model_parallel_size == 1.")
    if pp_degree <= 1:
        raise NotImplementedError("Only support dw_p2p_overlap=True if pipeline_model_parallel_size > 1.")


# ===========================================================================
# Muon slice math functions
# ===========================================================================


def _muon_qkv_per_head(matrix_2d_global, ortho_fn, num_query_groups=None, split_dims=None):
    """Slice fused QKV by groups, orthogonalise each head independently.

    ``split_dims`` is a list of 3 (Q, K, V) or 4 (Q, Gate, K, V) per-group widths.
    """
    import paddle

    groups = paddle.split(matrix_2d_global, num_query_groups, axis=1)
    q_heads_per_group = split_dims[0] // split_dims[-2]
    processed_groups = []
    if len(split_dims) == 4:
        for group in groups:
            q_part, gate_part, k_head, v_head = paddle.split(group, split_dims, axis=1)
            q_heads = paddle.split(q_part, q_heads_per_group, axis=1)
            q_ortho = paddle.concat([ortho_fn(h) for h in q_heads], axis=1)
            gate_heads = paddle.split(gate_part, q_heads_per_group, axis=1)
            gate_ortho = paddle.concat([ortho_fn(g) for g in gate_heads], axis=1)
            processed_groups.append(paddle.concat([q_ortho, gate_ortho, ortho_fn(k_head), ortho_fn(v_head)], axis=1))
    else:
        for group in groups:
            q_part, k_head, v_head = paddle.split(group, split_dims, axis=1)
            q_heads = paddle.split(q_part, q_heads_per_group, axis=1)
            q_ortho = paddle.concat([ortho_fn(h) for h in q_heads], axis=1)
            processed_groups.append(paddle.concat([q_ortho, ortho_fn(k_head), ortho_fn(v_head)], axis=1))
    return paddle.concat(processed_groups, axis=1)


def _muon_qkv_sep(matrix_2d, ortho_fn, num_query_groups=None, split_dims=None):
    """Slice fused QKV into Q (Gate) K V blocks, orthogonalise each as a whole."""
    import paddle

    groups = paddle.split(matrix_2d, num_query_groups, axis=1)
    q_parts, g_parts, k_parts, v_parts = [], [], [], []
    has_gate = len(split_dims) == 4
    for group in groups:
        if has_gate:
            q_p, g_p, k_p, v_p = paddle.split(group, split_dims, axis=1)
            g_parts.append(g_p)
        else:
            q_p, k_p, v_p = paddle.split(group, split_dims, axis=1)
        q_parts.append(q_p)
        k_parts.append(k_p)
        v_parts.append(v_p)

    q_ortho = ortho_fn(paddle.concat(q_parts, axis=1))
    g_ortho = ortho_fn(paddle.concat(g_parts, axis=1)) if has_gate else None
    k_ortho = ortho_fn(paddle.concat(k_parts, axis=1))
    v_ortho = ortho_fn(paddle.concat(v_parts, axis=1))

    q_groups = paddle.split(q_ortho, num_query_groups, axis=1)
    g_groups = paddle.split(g_ortho, num_query_groups, axis=1) if has_gate else None
    k_groups = paddle.split(k_ortho, num_query_groups, axis=1)
    v_groups = paddle.split(v_ortho, num_query_groups, axis=1)

    if has_gate:
        return paddle.concat(
            [
                paddle.concat([q_groups[i], g_groups[i], k_groups[i], v_groups[i]], axis=1)
                for i in range(num_query_groups)
            ],
            axis=1,
        )
    return paddle.concat(
        [paddle.concat([q_groups[i], k_groups[i], v_groups[i]], axis=1) for i in range(num_query_groups)],
        axis=1,
    )


def _muon_ffn_gate_up(matrix, ortho_fn, intermediate_size=None):
    """Slice FFN gate_up, orthogonalise gate and up independently.

    Handles both 2D (per-expert) and 3D (fused experts) weight tensors.
    For 3D tensors (e.g. [num_experts, hidden_size, 2*intermediate_size]),
    each expert slice is orthogonalised independently and re-stacked.
    """
    import paddle

    assert matrix.ndim == 2 or matrix.ndim == 3, "FFN gate_up split expects 2D or 3D tensor"

    if matrix.ndim == 3:
        expert_updates = []
        for ei in range(matrix.shape[0]):
            gate, up = paddle.split(matrix[ei], [intermediate_size, intermediate_size], axis=1)
            expert_updates.append(paddle.concat([ortho_fn(gate), ortho_fn(up)], axis=1))
        return paddle.stack(expert_updates, axis=0)

    gate, up = paddle.split(matrix, [intermediate_size, intermediate_size], axis=1)
    return paddle.concat([ortho_fn(gate), ortho_fn(up)], axis=1)


def _muon_mla_per_head(matrix_2d_global, ortho_fn, head_num=None, axis=None, head_split_sizes=None):
    """Slice MLA weights by heads."""
    import paddle

    split_args = head_num if head_split_sizes is None else head_split_sizes * head_num
    groups = paddle.split(matrix_2d_global, split_args, axis=axis)
    processed_groups = [ortho_fn(group) for group in groups]
    return paddle.concat(processed_groups, axis=axis)


def _muon_moe_experts(matrix_3d_global, ortho_fn):
    """Slice fused MoE weights by experts."""
    import paddle

    if matrix_3d_global.ndim != 3:
        raise ValueError(f"MoE expert split expects 3D tensor, got shape {matrix_3d_global.shape}")
    n_experts = matrix_3d_global.shape[0]
    return paddle.stack([ortho_fn(matrix_3d_global[ei]) for ei in range(n_experts)], axis=0)


# ===========================================================================
# Muon slice config builder
# ===========================================================================


def _build_muon_slice_config(model, config) -> dict:
    """Build declarative slice configuration for Muon optimizer."""
    slice_config = {}

    muon_configs = config.muon_configs

    num_hidden_layers = config.num_hidden_layers
    use_mla = bool(getattr(config, "multi_latent_attention", False))
    use_gated_attn = getattr(config, "use_gated_attn", False)
    moe_expert_fusion = getattr(config, "moe_expert_fusion", False)
    csa_compress_ratios = getattr(config, "csa_compress_ratios", None)
    use_vha_attention = getattr(config, "use_vha_attention", False)
    num_empty_layers_add_in_head = getattr(config, "num_empty_layers_add_in_head", 0)

    muon_qkv_update_mode = muon_configs.get("muon_qkv_update_mode", "split_head")
    muon_ffn_split = muon_configs.get("muon_ffn_split", False)

    qkv_slice_fn = None
    if muon_qkv_update_mode == "split_head":
        qkv_slice_fn = _muon_qkv_per_head
    elif muon_qkv_update_mode == "split_qkv":
        qkv_slice_fn = _muon_qkv_sep

    ffn_slice_fn = _muon_ffn_gate_up if muon_ffn_split else None
    fused_moe_fn = _muon_moe_experts if moe_expert_fusion else None

    # vha_premix_weight shape [num_kv_heads, head_dim, head_dim]; reuse 3D moe_experts func.
    vha_premix_fn = _muon_moe_experts if use_vha_attention else None

    mla_slice_fn = None
    if use_mla and muon_qkv_update_mode == "split_head":
        mla_slice_fn = _muon_mla_per_head

    def _add_layer_slice_config(prefix, layer_idx):
        # DSV4 hybrid attention (csa)
        if csa_compress_ratios is not None and mla_slice_fn is not None:
            real_layer_idx = layer_idx - num_empty_layers_add_in_head
            ratio = csa_compress_ratios[real_layer_idx]
            num_attention_heads = config.num_attention_heads
            slice_config[f"{prefix}.self_attn.linear_q_up_proj.weight"] = (
                mla_slice_fn,
                {"head_num": num_attention_heads, "axis": 1},
            )

            if ratio == 4:
                slice_config[f"{prefix}.self_attn.core_attention.compressor.linear_wkv.weight"] = (
                    mla_slice_fn,
                    {"head_num": 1, "axis": 1, "head_split_sizes": [config.v_head_dim, config.v_head_dim]},
                )
                slice_config[f"{prefix}.self_attn.core_attention.compressor.linear_wgate.weight"] = (
                    mla_slice_fn,
                    {"head_num": 1, "axis": 1, "head_split_sizes": [config.v_head_dim, config.v_head_dim]},
                )
            if ratio == 4 and getattr(config, "csa_dense_mode", False) is False:
                slice_config[f"{prefix}.self_attn.core_attention.indexer.linear_wq_b.weight"] = (
                    mla_slice_fn,
                    {"head_num": config.dsa_index_n_heads, "axis": 1},
                )
                slice_config[f"{prefix}.self_attn.core_attention.indexer.compressor.linear_wkv.weight"] = (
                    mla_slice_fn,
                    {
                        "head_num": 1,
                        "axis": 1,
                        "head_split_sizes": [config.dsa_index_head_dim, config.dsa_index_head_dim],
                    },
                )
                slice_config[f"{prefix}.self_attn.core_attention.indexer.compressor.linear_wgate.weight"] = (
                    mla_slice_fn,
                    {
                        "head_num": 1,
                        "axis": 1,
                        "head_split_sizes": [config.dsa_index_head_dim, config.dsa_index_head_dim],
                    },
                )

        num_heads, num_kv_heads, num_query_groups, split_dims, layer_is_swa = _get_layer_attn_split_info(
            config, layer_idx
        )

        # Fused QKV weights (non-MLA path)
        if not use_mla and qkv_slice_fn is not None:
            if use_vha_attention:
                # VHA: independent projections; split-by-head, ortho per head.
                if muon_qkv_update_mode == "split_head":
                    g = num_heads // num_kv_heads
                    q_lora_rank = config.swa_vha_q_lora_rank if layer_is_swa else config.vha_q_lora_rank
                    head_dim = config.swa_head_dim if layer_is_swa else config.head_dim
                    v_head_dim = config.swa_v_head_dim if layer_is_swa else config.v_head_dim
                    slice_config[f"{prefix}.self_attn.q_proj.weight"] = (
                        _muon_mla_per_head,
                        {"head_num": g, "axis": 1, "head_split_sizes": [q_lora_rank]},
                    )
                    slice_config[f"{prefix}.self_attn.k_proj.weight"] = (
                        _muon_mla_per_head,
                        {"head_num": num_kv_heads, "axis": 1, "head_split_sizes": [head_dim]},
                    )
                    slice_config[f"{prefix}.self_attn.v_proj.weight"] = (
                        _muon_mla_per_head,
                        {"head_num": num_kv_heads, "axis": 1, "head_split_sizes": [v_head_dim]},
                    )
                    if use_gated_attn:
                        slice_config[f"{prefix}.self_attn.gate_proj.weight"] = (
                            _muon_mla_per_head,
                            {"head_num": num_heads, "axis": 1, "head_split_sizes": [v_head_dim]},
                        )
                    if vha_premix_fn is not None:
                        slice_config[f"{prefix}.self_attn.vha_premix_weight"] = (vha_premix_fn, {})
                # split_qkv: no fuse, skip
            else:
                qkv_kwargs = {"num_query_groups": num_query_groups, "split_dims": split_dims}
                slice_config[f"{prefix}.self_attn.qkv_proj.weight"] = (qkv_slice_fn, qkv_kwargs)

        if ffn_slice_fn is not None:
            moe_intermediate_size = config.moe_intermediate_size
            intermediate_size = config.intermediate_size

            slice_config[f"{prefix}.mlp.experts.up_gate_proj.weight"] = (
                ffn_slice_fn,
                {"intermediate_size": moe_intermediate_size},
            )
            slice_config[f"{prefix}.mlp.shared_experts.up_gate_proj.weight"] = (
                ffn_slice_fn,
                {"intermediate_size": moe_intermediate_size},
            )
            slice_config[f"{prefix}.mlp.grouped_gemm_experts.weight1"] = (
                ffn_slice_fn,
                {"intermediate_size": moe_intermediate_size},
            )
            slice_config[f"{prefix}.mlp.up_gate_proj.weight"] = (
                ffn_slice_fn,
                {"intermediate_size": intermediate_size},
            )

            if hasattr(config, "n_routed_experts") and config.n_routed_experts > 0:
                for expert_idx in range(config.n_routed_experts):
                    slice_config[f"{prefix}.mlp.experts.{expert_idx}.up_gate_proj.weight"] = (
                        ffn_slice_fn,
                        {"intermediate_size": moe_intermediate_size},
                    )

        if moe_expert_fusion and fused_moe_fn is not None:
            slice_config[f"{prefix}.mlp.experts.down_proj.weight"] = (fused_moe_fn, {})
            slice_config[f"{prefix}.mlp.grouped_gemm_experts.weight2"] = (fused_moe_fn, {})

        if use_mla and mla_slice_fn is not None:
            # Reset num_attention_heads / v_head_dim for SWA layers.
            num_attention_heads = num_heads
            v_head_dim = split_dims[-1]
            assert (
                hasattr(config, "qk_nope_head_dim")
                and hasattr(config, "qk_rope_head_dim")
                and hasattr(config, "kv_lora_rank")
                and hasattr(config, "v_head_dim")
            )

            slice_config[f"{prefix}.self_attn.q_b_proj.weight"] = (
                mla_slice_fn,
                {
                    "head_num": num_attention_heads,
                    "axis": 1,
                    "head_split_sizes": [config.qk_nope_head_dim, config.qk_rope_head_dim],
                },
            )
            slice_config[f"{prefix}.self_attn.kv_a_proj_with_mqa.weight"] = (
                mla_slice_fn,
                {"head_num": 1, "axis": 1, "head_split_sizes": [config.kv_lora_rank, config.qk_rope_head_dim]},
            )
            slice_config[f"{prefix}.self_attn.kv_b_proj.weight"] = (
                mla_slice_fn,
                {
                    "head_num": num_attention_heads,
                    "axis": 1,
                    "head_split_sizes": [config.qk_nope_head_dim, v_head_dim],
                },
            )

            if use_gated_attn:
                slice_config[f"{prefix}.self_attn.gate_proj.weight"] = (
                    mla_slice_fn,
                    {"head_num": num_attention_heads, "axis": 1},
                )

    # Main layers (apply num_empty_layers_add_in_head offset)
    for layer_idx in range(num_hidden_layers):
        real_layer_number = layer_idx + num_empty_layers_add_in_head
        _add_layer_slice_config(f"model.layers.{real_layer_number}", real_layer_number)

    # MTP layers
    if getattr(config, "mtp_num_layers", 0) > 0:
        num_nextn_predict_layers = config.mtp_num_layers
    else:
        num_nextn_predict_layers = config.num_nextn_predict_layers if config.num_nextn_predict_layers else 0
    for layer_idx in range(num_nextn_predict_layers):
        real_layer_number = layer_idx + num_empty_layers_add_in_head + num_hidden_layers
        _add_layer_slice_config(f"model.layers.{real_layer_number}", real_layer_number)
    for layer_idx in range(num_nextn_predict_layers):
        real_layer_number = layer_idx + num_empty_layers_add_in_head + num_hidden_layers
        _add_layer_slice_config(f"model.layers.{real_layer_number}.transformer_layer", real_layer_number)

    return slice_config


def build_muon_param_info_map(model, config):
    """Build parameter info map for Muon optimizer."""
    from paddle.optimizer.muon import MuonParamInfo, _default_should_use_muon

    info_map = {}
    exclude_patterns = config.muon_configs["muon_exclude_patterns"]

    slice_config = _build_muon_slice_config(model, config)

    pp_to_single = getattr(model, "_pp_to_single_mapping", None)
    if pp_to_single is None:
        try:
            single_to_pp = model._set_pipeline_name_mapping()
            if single_to_pp:
                pp_to_single = {v: k for k, v in single_to_pp.items()}
        except Exception as e:
            logger.warning(f"_set_pipeline_name_mapping failed: {e}")
    if pp_to_single is None:
        pp_to_single = {}

    for pp_name, param in model.named_parameters():
        name = pp_to_single.get(pp_name, pp_name)
        use_muon = _default_should_use_muon(name, param.shape, exclude_patterns) and _default_should_use_muon(
            param.name, param.shape, exclude_patterns
        )

        if name in slice_config:
            slice_fn, slice_kwargs = slice_config[name]
            split_concat_func = partial(slice_fn, **slice_kwargs)
        else:
            split_concat_func = None

        param_info = MuonParamInfo(
            use_muon=use_muon,
            split_concat_func=split_concat_func,
        )

        info_map[param.name] = param_info

        sc_func = param_info.split_concat_func
        func_name = sc_func.func.__name__ if sc_func else None
        func_kwargs = sc_func.keywords if sc_func else {}

        logger.info(
            f"name: {name}, param.name: {param.name}, shape: {param.shape}, "
            f"use_muon: {use_muon}, "
            f"split_concat_func: {func_name}, "
            f"split_concat_func_kwargs: {func_kwargs}"
        )

    return info_map


# ===========================================================================
# AOA config generators
# ===========================================================================


def _gen_aoa_config(config):
    """Generate AOA config for loading HF-format weights into Fleet model."""
    model_prefix = "model."
    using_sonic_moe = getattr(config, "using_sonic_moe", False)
    if hasattr(config, "n_routed_experts"):
        num_experts = config.n_routed_experts
    else:
        num_experts = config.num_experts

    aoa_config = {
        "aoa_statements": [
            f"model.norm.weight -> {model_prefix}norm.weight",
        ]
    }

    aoa_config["aoa_statements"] += [
        f"model.embed_tokens.weight -> {model_prefix}embedding.embed_tokens.weight",
    ]

    assert not (
        config.tie_word_embeddings and getattr(config, "separate_mtp_headloss", False)
    ), "tie_word_embeddings and separate_mtp_headloss cannot be enabled simultaneously in aoa"
    if config.tie_word_embeddings:
        aoa_config["aoa_statements"] += [f"model.embed_tokens.weight -> {model_prefix}lm_head.weight"]
    elif getattr(config, "separate_mtp_headloss", False):
        aoa_config["aoa_statements"] += [f"lm_head.weight -> {model_prefix}shared_mtp_lm_head.weight"]
        aoa_config["aoa_statements"] += [f"lm_head.weight -> {model_prefix}shared_head.weight"]
    else:
        aoa_config["aoa_statements"] += [f"lm_head.weight -> {model_prefix}lm_head.weight"]

    num_hidden_layers = config.num_hidden_layers
    num_head_empty_layers = (
        config.num_empty_layers_add_in_head
        if hasattr(config, "num_empty_layers_add_in_head") and config.num_empty_layers_add_in_head
        else 0
    )

    if getattr(config, "mtp_num_layers", 0) > 0:
        num_nextn_predict_layers = config.mtp_num_layers
    else:
        num_nextn_predict_layers = config.num_nextn_predict_layers if config.num_nextn_predict_layers else 0

    n_shared_experts = config.n_shared_experts if hasattr(config, "n_shared_experts") else 0
    if n_shared_experts > 0:
        assert n_shared_experts == 1, f"n_shared_experts must be 0 or 1, but got {n_shared_experts}"

    # MTP layers
    for layer_idx in reversed(range(num_hidden_layers, num_hidden_layers + num_nextn_predict_layers)):
        layer_idx_offset = layer_idx + num_head_empty_layers
        prefix = f"model.layers.{layer_idx}"
        prefix_offset = f"{model_prefix}layers.{layer_idx_offset}"
        aoa_config["aoa_statements"] += [
            f"{prefix}.eh_proj.weight^T -> {prefix_offset}.eh_proj.weight",
            f"{prefix}.enorm.weight -> {prefix_offset}.enorm.weight",
            f"{prefix}.hnorm.weight -> {prefix_offset}.hnorm.weight",
            f"{prefix}.shared_head.norm.weight -> {prefix_offset}.norm.weight",
        ]

        if config.use_dense_mtp:
            prefix_offset += ".transformer_layer"
            aoa_config["aoa_statements"] += [
                f"{prefix}.mlp.gate_proj.weight^T, "
                f"{prefix}.mlp.up_proj.weight^T -> "
                f"{prefix_offset}.mlp.up_gate_proj.weight, fused_ffn",
                f"{prefix}.mlp.down_proj.weight^T -> {prefix_offset}.mlp.down_proj.weight",
            ]

    # Transformer layers
    for layer_idx in reversed(range(0, num_hidden_layers + num_nextn_predict_layers)):
        layer_idx_offset = layer_idx + num_head_empty_layers
        prefix = f"model.layers.{layer_idx}"
        prefix_offset = f"{model_prefix}layers.{layer_idx_offset}"
        if layer_idx >= num_hidden_layers:
            prefix_offset += ".transformer_layer"
        aoa_config["aoa_statements"] += [
            f"{prefix}.input_layernorm.weight -> {prefix_offset}.input_layernorm.weight",
            f"{prefix}.post_attention_layernorm.weight -> {prefix_offset}.post_attention_layernorm.weight",
            f"{prefix}.self_attn.o_proj.weight^T -> {prefix_offset}.self_attn.o_proj.weight",
        ]

        use_mla = bool(getattr(config, "multi_latent_attention", False))

        if config.use_gated_attn and use_mla:
            # MLA mode: gate_proj is a separate parameter
            aoa_config["aoa_statements"] += [
                f"{prefix}.self_attn.gate_proj.weight^T -> {prefix_offset}.self_attn.gate_proj.weight",
            ]

        is_swa = is_layer_window_attention(
            getattr(config, "sliding_window", None),
            getattr(config, "window_attn_skip_freq", None),
            layer_idx,
        )
        if (
            getattr(config, "softmax_type", None) == "learnable"
            or (getattr(config, "add_full_attention_sink_bias", False) and not is_swa)
            or (getattr(config, "add_swa_attention_sink_bias", False) and is_swa)
        ):
            aoa_config["aoa_statements"] += [
                f"{prefix}.self_attn.core_attention.softmax_offset -> "
                f"{prefix_offset}.self_attn.core_attention.softmax_offset",
            ]

        if getattr(config, "use_vha_attention", False):
            aoa_config["aoa_statements"] += [
                f"{prefix}.self_attn.vha_premix_weight -> {prefix_offset}.self_attn.vha_premix_weight",
                f"{prefix}.self_attn.vha_postmix_U -> {prefix_offset}.self_attn.vha_postmix_U",
                f"{prefix}.self_attn.vha_postmix_V -> {prefix_offset}.self_attn.vha_postmix_V",
            ]

        if use_mla:
            aoa_config["aoa_statements"] += [
                f"{prefix}.self_attn.q_a_proj.weight^T -> {prefix_offset}.self_attn.q_a_proj.weight",
                f"{prefix}.self_attn.q_b_proj.weight^T -> {prefix_offset}.self_attn.q_b_proj.weight",
                f"{prefix}.self_attn.kv_a_proj_with_mqa.weight^T -> "
                f"{prefix_offset}.self_attn.kv_a_proj_with_mqa.weight",
                f"{prefix}.self_attn.kv_b_proj.weight^T -> {prefix_offset}.self_attn.kv_b_proj.weight",
            ]
            if config.use_qk_norm:
                aoa_config["aoa_statements"] += [
                    f"{prefix}.self_attn.q_a_layernorm.weight -> {prefix_offset}.self_attn.q_a_layernorm.weight",
                    f"{prefix}.self_attn.kv_a_layernorm.weight -> {prefix_offset}.self_attn.kv_a_layernorm.weight",
                ]
        elif getattr(config, "experimental_attention_variant", None) == "dsv4_hybrid":
            # csa_compress_ratios covers main + MTP layers.
            assert len(config.csa_compress_ratios) == num_hidden_layers + num_nextn_predict_layers, (
                f"csa_compress_ratios length ({len(config.csa_compress_ratios)}) must equal "
                f"num_hidden_layers + num_nextn_predict_layers "
                f"({num_hidden_layers} + {num_nextn_predict_layers})"
            )
            csa_ratio = config.csa_compress_ratios[layer_idx]
            aoa_config["aoa_statements"] += [
                f"{prefix}.self_attn.linear_q_down_proj.weight^T -> "
                f"{prefix_offset}.self_attn.linear_q_down_proj.weight",
                f"{prefix}.self_attn.linear_q_up_proj.weight^T -> "
                f"{prefix_offset}.self_attn.linear_q_up_proj.weight",
                f"{prefix}.self_attn.linear_kv_proj.weight^T -> " f"{prefix_offset}.self_attn.linear_kv_proj.weight",
                f"{prefix}.self_attn.o_proj.weight^T -> {prefix_offset}.self_attn.o_proj.weight",
                f"{prefix}.self_attn.q_layernorm.weight -> {prefix_offset}.self_attn.q_layernorm.weight",
                f"{prefix}.self_attn.kv_layernorm.weight -> {prefix_offset}.self_attn.kv_layernorm.weight",
                f"{prefix}.self_attn.linear_o_group_proj -> {prefix_offset}.self_attn.linear_o_group_proj",
                f"{prefix}.self_attn.core_attention.attn_sink -> "
                f"{prefix_offset}.self_attn.core_attention.attn_sink",
            ]
            if csa_ratio > 1:
                aoa_config["aoa_statements"] += [
                    f"{prefix}.self_attn.core_attention.compressor.linear_wkv.weight^T -> "
                    f"{prefix_offset}.self_attn.core_attention.compressor.linear_wkv.weight",
                    f"{prefix}.self_attn.core_attention.compressor.linear_wgate.weight^T -> "
                    f"{prefix_offset}.self_attn.core_attention.compressor.linear_wgate.weight",
                    f"{prefix}.self_attn.core_attention.compressor.norm.weight -> "
                    f"{prefix_offset}.self_attn.core_attention.compressor.norm.weight",
                    f"{prefix}.self_attn.core_attention.compressor.ape -> "
                    f"{prefix_offset}.self_attn.core_attention.compressor.ape",
                ]
            if csa_ratio == 4 and not getattr(config, "csa_dense_mode", False):
                aoa_config["aoa_statements"] += [
                    f"{prefix}.self_attn.core_attention.indexer.linear_wq_b.weight^T -> "
                    f"{prefix_offset}.self_attn.core_attention.indexer.linear_wq_b.weight",
                    f"{prefix}.self_attn.core_attention.indexer.linear_weights_proj.weight^T -> "
                    f"{prefix_offset}.self_attn.core_attention.indexer.linear_weights_proj.weight",
                    f"{prefix}.self_attn.core_attention.indexer.compressor.linear_wkv.weight^T -> "
                    f"{prefix_offset}.self_attn.core_attention.indexer.compressor.linear_wkv.weight",
                    f"{prefix}.self_attn.core_attention.indexer.compressor.linear_wgate.weight^T -> "
                    f"{prefix_offset}.self_attn.core_attention.indexer.compressor.linear_wgate.weight",
                    f"{prefix}.self_attn.core_attention.indexer.compressor.norm.weight -> "
                    f"{prefix_offset}.self_attn.core_attention.indexer.compressor.norm.weight",
                    f"{prefix}.self_attn.core_attention.indexer.compressor.ape -> "
                    f"{prefix_offset}.self_attn.core_attention.indexer.compressor.ape",
                ]
        else:
            if config.use_qk_norm:
                aoa_config["aoa_statements"] += [
                    f"{prefix}.self_attn.q_norm.weight -> {prefix_offset}.self_attn.q_norm.weight",
                    f"{prefix}.self_attn.k_norm.weight -> {prefix_offset}.self_attn.k_norm.weight",
                ]

            # Attention QKV (SWA-aware split info uses real layer index)
            num_heads, num_kv_heads, num_query_groups, split_dims, layer_is_swa = _get_layer_attn_split_info(
                config, layer_idx + num_head_empty_layers
            )

            if len(split_dims) == 4:
                q_dim, gate_dim, head_dim, v_head_dim = split_dims
                use_gated_attn_layer = True
            else:
                q_dim, head_dim, v_head_dim = split_dims
                use_gated_attn_layer = False

            gcd_head_dim = math.gcd(head_dim, v_head_dim)
            head_dim_chunks = head_dim // gcd_head_dim
            v_head_dim_chunks = v_head_dim // gcd_head_dim

            if getattr(config, "use_vha_attention", False):
                aoa_config["aoa_statements"] += [
                    f"{prefix}.self_attn.q_proj.weight^T -> {prefix_offset}.self_attn.q_proj.weight",
                    f"{prefix}.self_attn.k_proj.weight^T -> {prefix_offset}.self_attn.k_proj.weight",
                    f"{prefix}.self_attn.v_proj.weight^T -> {prefix_offset}.self_attn.v_proj.weight",
                ]
                if use_gated_attn_layer:
                    aoa_config["aoa_statements"].append(
                        f"{prefix}.self_attn.gate_proj.weight^T -> {prefix_offset}.self_attn.gate_proj.weight"
                    )
                if config.attention_bias:
                    aoa_config["aoa_statements"] += [
                        f"{prefix}.self_attn.q_proj.bias -> {prefix_offset}.self_attn.q_proj.bias",
                        f"{prefix}.self_attn.k_proj.bias -> {prefix_offset}.self_attn.k_proj.bias",
                        f"{prefix}.self_attn.v_proj.bias -> {prefix_offset}.self_attn.v_proj.bias",
                    ]
                    if use_gated_attn_layer:
                        aoa_config["aoa_statements"].append(
                            f"{prefix}.self_attn.gate_proj.bias -> {prefix_offset}.self_attn.gate_proj.bias"
                        )
            else:
                # Non-VHA: split per-head, possibly with chunking when head_dim != v_head_dim,
                # then re-fuse Fleet layout per group: [Q_heads, Gate_heads, K, V].
                q_names = []
                gate_names = []
                k_names = []
                v_names = []
                q_bias_names = []
                gate_bias_names = []
                k_bias_names = []
                v_bias_names = []
                heads_per_group = num_heads // num_kv_heads
                for g in range(num_kv_heads):
                    for h in range(heads_per_group):
                        for c in range(head_dim_chunks):
                            q_names.append(f"{prefix}.self_attn._q_g{g}_h{h}_c{c}")
                            q_bias_names.append(f"{prefix}.self_attn._q_g{g}_h{h}_c{c}_bias")
                        if use_gated_attn_layer:
                            for c in range(v_head_dim_chunks):
                                gate_names.append(f"{prefix}.self_attn._gate_g{g}_h{h}_c{c}")
                                gate_bias_names.append(f"{prefix}.self_attn._gate_g{g}_h{h}_c{c}_bias")

                    for c in range(head_dim_chunks):
                        k_names.append(f"{prefix}.self_attn._k_g{g}_c{c}")
                        k_bias_names.append(f"{prefix}.self_attn._k_g{g}_c{c}_bias")
                    for c in range(v_head_dim_chunks):
                        v_names.append(f"{prefix}.self_attn._v_g{g}_c{c}")
                        v_bias_names.append(f"{prefix}.self_attn._v_g{g}_c{c}_bias")

                aoa_config["aoa_statements"].append(f"{prefix}.self_attn.q_proj.weight -> {','.join(q_names)}, axis=0")
                if use_gated_attn_layer:
                    aoa_config["aoa_statements"].append(
                        f"{prefix}.self_attn.gate_proj.weight -> {','.join(gate_names)}, axis=0"
                    )
                aoa_config["aoa_statements"].append(f"{prefix}.self_attn.k_proj.weight -> {','.join(k_names)}, axis=0")
                aoa_config["aoa_statements"].append(f"{prefix}.self_attn.v_proj.weight -> {','.join(v_names)}, axis=0")

                ordered = []
                bias_ordered = []
                for g in range(num_kv_heads):
                    for h in range(heads_per_group):
                        for c in range(head_dim_chunks):
                            ordered.append(f"{prefix}.self_attn._q_g{g}_h{h}_c{c}")
                            bias_ordered.append(f"{prefix}.self_attn._q_g{g}_h{h}_c{c}_bias")

                    if use_gated_attn_layer:
                        for h in range(heads_per_group):
                            for c in range(v_head_dim_chunks):
                                ordered.append(f"{prefix}.self_attn._gate_g{g}_h{h}_c{c}")
                                bias_ordered.append(f"{prefix}.self_attn._gate_g{g}_h{h}_c{c}_bias")

                    for c in range(head_dim_chunks):
                        ordered.append(f"{prefix}.self_attn._k_g{g}_c{c}")
                        bias_ordered.append(f"{prefix}.self_attn._k_g{g}_c{c}_bias")
                    for c in range(v_head_dim_chunks):
                        ordered.append(f"{prefix}.self_attn._v_g{g}_c{c}")
                        bias_ordered.append(f"{prefix}.self_attn._v_g{g}_c{c}_bias")

                fused_tmp = f"{prefix}.self_attn.qkv_fused_tmp"
                aoa_config["aoa_statements"].append(f"{','.join(ordered)} -> {fused_tmp}, axis=0")
                aoa_config["aoa_statements"].append(f"{fused_tmp}^T -> {prefix_offset}.self_attn.qkv_proj.weight")

                if config.attention_bias:
                    aoa_config["aoa_statements"].append(
                        f"{prefix}.self_attn.q_proj.bias -> {','.join(q_bias_names)}, axis=0"
                    )
                    if use_gated_attn_layer:
                        aoa_config["aoa_statements"].append(
                            f"{prefix}.self_attn.gate_proj.bias -> {','.join(gate_bias_names)}, axis=0"
                        )
                    aoa_config["aoa_statements"].append(
                        f"{prefix}.self_attn.k_proj.bias -> {','.join(k_bias_names)}, axis=0"
                    )
                    aoa_config["aoa_statements"].append(
                        f"{prefix}.self_attn.v_proj.bias -> {','.join(v_bias_names)}, axis=0"
                    )
                    aoa_config["aoa_statements"].append(
                        f"{','.join(bias_ordered)} -> {prefix_offset}.self_attn.qkv_proj.bias, axis=0"
                    )

    # MoE gate / expert weight mapping
    moe_layer_start = config.first_k_dense_replace
    moe_layer_end = num_hidden_layers if config.use_dense_mtp else num_hidden_layers + num_nextn_predict_layers
    for layer_idx in reversed(range(moe_layer_start, moe_layer_end)):
        layer_idx_offset = layer_idx + num_head_empty_layers
        prefix = f"model.layers.{layer_idx}"
        prefix_offset = f"{model_prefix}layers.{layer_idx_offset}"
        if layer_idx >= num_hidden_layers:
            prefix_offset += ".transformer_layer"
        aoa_config["aoa_statements"] += [
            f"{prefix}.block_sparse_moe.e_score_correction_bias -> {prefix_offset}.mlp.gate.e_score_correction_bias",
            f"{prefix}.block_sparse_moe.gate.weight -> {prefix_offset}.mlp.gate.weight",
        ]
        if getattr(config, "routed_scaling_factor_learnable", False):
            aoa_config["aoa_statements"] += [
                f"{prefix}.block_sparse_moe.gate.routed_scaling_factor_param -> "
                f"{prefix_offset}.mlp.gate.routed_scaling_factor_param",
            ]
        if config.moe_latent_size is not None and config.moe_latent_size > 0:
            aoa_config["aoa_statements"] += [
                f"{prefix}.block_sparse_moe.fc1_latent_proj.weight^T -> {prefix_offset}.mlp.fc1_latent_proj.weight",
                f"{prefix}.block_sparse_moe.fc2_latent_proj.weight^T -> {prefix_offset}.mlp.fc2_latent_proj.weight",
            ]

        if using_sonic_moe:
            aoa_config["aoa_statements"] += [
                f"{prefix}.block_sparse_moe.experts.$EXPERT_ID.w2.weight -> "
                f"{prefix_offset}.mlp.experts.$EXPERT_ID.down_proj.weight",
            ]
        else:
            aoa_config["aoa_statements"] += [
                f"{prefix}.block_sparse_moe.experts.$EXPERT_ID.w2.weight^T -> "
                f"{prefix_offset}.mlp.experts.$EXPERT_ID.down_proj.weight",
            ]

        if n_shared_experts > 0:
            aoa_config["aoa_statements"] += [
                f"{prefix}.block_sparse_moe.shared_experts.w1.weight^T, "
                f"{prefix}.block_sparse_moe.shared_experts.w3.weight^T -> "
                f"{prefix_offset}.mlp.shared_experts.up_gate_proj.weight, fused_ffn",
                f"{prefix}.block_sparse_moe.shared_experts.w2.weight^T -> "
                f"{prefix_offset}.mlp.shared_experts.down_proj.weight",
            ]

        for expert_id in range(config.n_routed_experts):
            if using_sonic_moe:
                aoa_config["aoa_statements"] += [
                    f"{prefix}.block_sparse_moe.experts.{expert_id}.w1.weight, "
                    f"{prefix}.block_sparse_moe.experts.{expert_id}.w3.weight -> "
                    f"{prefix_offset}.mlp.experts.{expert_id}.up_gate_proj.weight, axis=0",
                ]
            else:
                aoa_config["aoa_statements"] += [
                    f"{prefix}.block_sparse_moe.experts.{expert_id}.w1.weight^T, "
                    f"{prefix}.block_sparse_moe.experts.{expert_id}.w3.weight^T -> "
                    f"{prefix_offset}.mlp.experts.{expert_id}.up_gate_proj.weight, axis=1",
                ]

        use_fused_weight = config.moe_expert_fusion
        if (
            getattr(config, "fp8", False)
            and (config.moe_expert_fusion is False)
            and getattr(config, "moe_deep_gemm", False)
        ):
            import logging

            logging.getLogger(__name__).warning(
                "For fp8 deep_gemm, moe_expert_fusion should be True. Skipping fused weight in inv_aoa."
            )
            use_fused_weight = False
        if (
            getattr(config, "fp8", False)
            and config.moe_expert_fusion
            and getattr(config, "moe_deep_gemm", None) is False
        ):
            use_fused_weight = False

        if use_fused_weight:
            ep_weight1 = []
            ep_weight2 = []
            for expert_id in range(num_experts):
                ep_weight1.append(f"{prefix}.mlp.experts.{expert_id}.up_gate_proj.weight")
                ep_weight2.append(f"{prefix}.mlp.experts.{expert_id}.down_proj.weight")
            group_gemm1 = ",".join(ep_weight1)
            group_gemm2 = ",".join(ep_weight2)
            aoa_config["aoa_statements"] += [
                f"{group_gemm1} -> {prefix_offset}.mlp.grouped_gemm_experts.weight1, axis=0",
                f"{group_gemm2} -> {prefix_offset}.mlp.grouped_gemm_experts.weight2, axis=0",
            ]
        else:
            if getattr(config, "fd_fallback", False):
                ep_weight1 = []
                ep_weight2 = []
                for expert_id in range(num_experts):
                    ep_weight1.append(f"{prefix_offset}.mlp.experts.{expert_id}.up_gate_proj.weight")
                    ep_weight2.append(f"{prefix_offset}.mlp.experts.{expert_id}.down_proj.weight")
                group1 = ",".join(ep_weight1)
                group2 = ",".join(ep_weight2)
                aoa_config["aoa_statements"] += [
                    f"{group1} -> {prefix_offset}.mlp.experts.up_gate_proj, axis=0",
                    f"{group2} -> {prefix_offset}.mlp.experts.down_proj, axis=0",
                ]

    return aoa_config


def _gen_inv_aoa_config(config):
    """Generate inverse AOA config for saving Fleet weights back to HF format."""
    model_prefix = "model."
    using_sonic_moe = getattr(config, "using_sonic_moe", False)
    if hasattr(config, "n_routed_experts"):
        num_experts = config.n_routed_experts
    else:
        num_experts = config.num_experts

    aoa_statements = [
        f"{model_prefix}norm.weight -> model.norm.weight",
    ]

    aoa_statements += [
        "model.embedding.embed_tokens.weight -> model.embed_tokens.weight",
    ]

    assert not (
        config.tie_word_embeddings and getattr(config, "separate_mtp_headloss", False)
    ), "tie_word_embeddings and separate_mtp_headloss cannot be enabled simultaneously in aoa"
    if config.tie_word_embeddings:
        aoa_statements += [f"{model_prefix}lm_head.weight -> _"]
    elif getattr(config, "separate_mtp_headloss", False):
        aoa_statements += [f"{model_prefix}shared_mtp_lm_head.weight -> lm_head.weight"]
        aoa_statements += [f"{model_prefix}shared_head.weight -> _"]
    else:
        aoa_statements += [f"{model_prefix}lm_head.weight -> lm_head.weight"]

    num_hidden_layers = config.num_hidden_layers
    num_head_empty_layers = (
        config.num_empty_layers_add_in_head
        if hasattr(config, "num_empty_layers_add_in_head") and config.num_empty_layers_add_in_head
        else 0
    )

    if getattr(config, "mtp_num_layers", 0) > 0:
        num_nextn_predict_layers = config.mtp_num_layers
    else:
        num_nextn_predict_layers = config.num_nextn_predict_layers if config.num_nextn_predict_layers else 0

    n_shared_experts = config.n_shared_experts if hasattr(config, "n_shared_experts") else 0
    if n_shared_experts > 0:
        assert n_shared_experts == 1, f"n_shared_experts must be 0 or 1, but got {n_shared_experts}"

    # MTP layers
    for layer_idx in reversed(range(num_hidden_layers, num_hidden_layers + num_nextn_predict_layers)):
        layer_idx_offset = layer_idx + num_head_empty_layers
        prefix = f"model.layers.{layer_idx}"
        prefix_offset = f"{model_prefix}layers.{layer_idx_offset}"
        aoa_statements += [
            f"{prefix_offset}.eh_proj.weight^T -> {prefix}.eh_proj.weight",
            f"{prefix_offset}.enorm.weight -> {prefix}.enorm.weight",
            f"{prefix_offset}.hnorm.weight -> {prefix}.hnorm.weight",
            f"{prefix_offset}.norm.weight -> {prefix}.shared_head.norm.weight",
        ]

        if config.use_dense_mtp:
            prefix_offset_tf = f"{prefix_offset}.transformer_layer"
            aoa_statements += [
                f"{prefix_offset_tf}.mlp.up_gate_proj.weight -> "
                f"{prefix}.mlp.gate_proj.weight, "
                f"{prefix}.mlp.up_proj.weight, fused_ffn",
                f"{prefix}.mlp.gate_proj.weight^T -> {prefix}.mlp.gate_proj.weight",
                f"{prefix}.mlp.up_proj.weight^T -> {prefix}.mlp.up_proj.weight",
                f"{prefix_offset_tf}.mlp.down_proj.weight^T -> {prefix}.mlp.down_proj.weight",
            ]

    # Transformer layers
    for layer_idx in range(0, num_hidden_layers + num_nextn_predict_layers):
        layer_idx_offset = layer_idx + num_head_empty_layers
        prefix_offset = f"{model_prefix}layers.{layer_idx_offset}"
        prefix = f"model.layers.{layer_idx}"
        if layer_idx >= num_hidden_layers:
            prefix_offset += ".transformer_layer"

        aoa_statements += [
            f"{prefix_offset}.input_layernorm.weight -> {prefix}.input_layernorm.weight",
            f"{prefix_offset}.post_attention_layernorm.weight -> {prefix}.post_attention_layernorm.weight",
            f"{prefix_offset}.self_attn.o_proj.weight^T -> {prefix}.self_attn.o_proj.weight",
        ]

        use_mla = bool(getattr(config, "multi_latent_attention", False))
        is_swa = is_layer_window_attention(
            getattr(config, "sliding_window", None),
            getattr(config, "window_attn_skip_freq", None),
            layer_idx,
        )

        if config.use_gated_attn and use_mla:
            # MLA mode: gate_proj is a separate parameter
            aoa_statements += [
                f"{prefix_offset}.self_attn.gate_proj.weight^T -> {prefix}.self_attn.gate_proj.weight",
            ]

        if getattr(config, "use_vha_attention", False):
            aoa_statements += [
                f"{prefix_offset}.self_attn.vha_premix_weight -> {prefix}.self_attn.vha_premix_weight",
                f"{prefix_offset}.self_attn.vha_postmix_U -> {prefix}.self_attn.vha_postmix_U",
                f"{prefix_offset}.self_attn.vha_postmix_V -> {prefix}.self_attn.vha_postmix_V",
            ]

        if (
            getattr(config, "softmax_type", None) == "learnable"
            or (getattr(config, "add_full_attention_sink_bias", False) and not is_swa)
            or (getattr(config, "add_swa_attention_sink_bias", False) and is_swa)
        ):
            aoa_statements += [
                f"{prefix_offset}.self_attn.core_attention.softmax_offset -> "
                f"{prefix}.self_attn.core_attention.softmax_offset",
            ]

        if use_mla:
            aoa_statements += [
                f"{prefix_offset}.self_attn.q_a_proj.weight^T -> {prefix}.self_attn.q_a_proj.weight",
                f"{prefix_offset}.self_attn.q_b_proj.weight^T -> {prefix}.self_attn.q_b_proj.weight",
                f"{prefix_offset}.self_attn.kv_a_proj_with_mqa.weight^T -> "
                f"{prefix}.self_attn.kv_a_proj_with_mqa.weight",
                f"{prefix_offset}.self_attn.kv_b_proj.weight^T -> {prefix}.self_attn.kv_b_proj.weight",
            ]
            if config.use_qk_norm:
                aoa_statements += [
                    f"{prefix_offset}.self_attn.q_a_layernorm.weight -> {prefix}.self_attn.q_a_layernorm.weight",
                    f"{prefix_offset}.self_attn.kv_a_layernorm.weight -> {prefix}.self_attn.kv_a_layernorm.weight",
                ]
        elif getattr(config, "experimental_attention_variant", None) == "dsv4_hybrid":
            assert len(config.csa_compress_ratios) == num_hidden_layers + num_nextn_predict_layers, (
                f"csa_compress_ratios length ({len(config.csa_compress_ratios)}) must equal "
                f"num_hidden_layers + num_nextn_predict_layers "
                f"({num_hidden_layers} + {num_nextn_predict_layers})"
            )
            csa_ratio = config.csa_compress_ratios[layer_idx]
            aoa_statements += [
                f"{prefix_offset}.self_attn.linear_q_down_proj.weight^T -> "
                f"{prefix}.self_attn.linear_q_down_proj.weight",
                f"{prefix_offset}.self_attn.linear_q_up_proj.weight^T -> "
                f"{prefix}.self_attn.linear_q_up_proj.weight",
                f"{prefix_offset}.self_attn.linear_kv_proj.weight^T -> " f"{prefix}.self_attn.linear_kv_proj.weight",
                f"{prefix_offset}.self_attn.o_proj.weight^T -> {prefix}.self_attn.o_proj.weight",
                f"{prefix_offset}.self_attn.q_layernorm.weight -> {prefix}.self_attn.q_layernorm.weight",
                f"{prefix_offset}.self_attn.kv_layernorm.weight -> {prefix}.self_attn.kv_layernorm.weight",
                f"{prefix_offset}.self_attn.linear_o_group_proj -> {prefix}.self_attn.linear_o_group_proj",
                f"{prefix_offset}.self_attn.core_attention.attn_sink -> "
                f"{prefix}.self_attn.core_attention.attn_sink",
            ]
            if csa_ratio > 1:
                aoa_statements += [
                    f"{prefix_offset}.self_attn.core_attention.compressor.linear_wkv.weight^T -> "
                    f"{prefix}.self_attn.core_attention.compressor.linear_wkv.weight",
                    f"{prefix_offset}.self_attn.core_attention.compressor.linear_wgate.weight^T -> "
                    f"{prefix}.self_attn.core_attention.compressor.linear_wgate.weight",
                    f"{prefix_offset}.self_attn.core_attention.compressor.norm.weight -> "
                    f"{prefix}.self_attn.core_attention.compressor.norm.weight",
                    f"{prefix_offset}.self_attn.core_attention.compressor.ape -> "
                    f"{prefix}.self_attn.core_attention.compressor.ape",
                ]
            if csa_ratio == 4 and not getattr(config, "csa_dense_mode", False):
                aoa_statements += [
                    f"{prefix_offset}.self_attn.core_attention.indexer.linear_wq_b.weight^T -> "
                    f"{prefix}.self_attn.core_attention.indexer.linear_wq_b.weight",
                    f"{prefix_offset}.self_attn.core_attention.indexer.linear_weights_proj.weight^T -> "
                    f"{prefix}.self_attn.core_attention.indexer.linear_weights_proj.weight",
                    f"{prefix_offset}.self_attn.core_attention.indexer.compressor.linear_wkv.weight^T -> "
                    f"{prefix}.self_attn.core_attention.indexer.compressor.linear_wkv.weight",
                    f"{prefix_offset}.self_attn.core_attention.indexer.compressor.linear_wgate.weight^T -> "
                    f"{prefix}.self_attn.core_attention.indexer.compressor.linear_wgate.weight",
                    f"{prefix_offset}.self_attn.core_attention.indexer.compressor.norm.weight -> "
                    f"{prefix}.self_attn.core_attention.indexer.compressor.norm.weight",
                    f"{prefix_offset}.self_attn.core_attention.indexer.compressor.ape -> "
                    f"{prefix}.self_attn.core_attention.indexer.compressor.ape",
                ]
        else:
            if config.use_qk_norm:
                aoa_statements += [
                    f"{prefix_offset}.self_attn.q_norm.weight -> {prefix}.self_attn.q_norm.weight",
                    f"{prefix_offset}.self_attn.k_norm.weight -> {prefix}.self_attn.k_norm.weight",
                ]

            num_heads, num_kv_heads, num_query_groups, split_dims, layer_is_swa = _get_layer_attn_split_info(
                config, layer_idx + num_head_empty_layers
            )

            if len(split_dims) == 4:
                q_dim, gate_dim, head_dim, v_head_dim = split_dims
                use_gated_attn_layer = True
            else:
                q_dim, head_dim, v_head_dim = split_dims
                use_gated_attn_layer = False

            gcd_head_dim = math.gcd(head_dim, v_head_dim)
            head_dim_chunks = head_dim // gcd_head_dim
            v_head_dim_chunks = v_head_dim // gcd_head_dim

            if getattr(config, "use_vha_attention", False):
                aoa_statements += [
                    f"{prefix_offset}.self_attn.q_proj.weight^T -> {prefix}.self_attn.q_proj.weight",
                    f"{prefix_offset}.self_attn.k_proj.weight^T -> {prefix}.self_attn.k_proj.weight",
                    f"{prefix_offset}.self_attn.v_proj.weight^T -> {prefix}.self_attn.v_proj.weight",
                ]
                if use_gated_attn_layer:
                    aoa_statements.append(
                        f"{prefix_offset}.self_attn.gate_proj.weight^T -> {prefix}.self_attn.gate_proj.weight"
                    )
                if config.attention_bias:
                    aoa_statements += [
                        f"{prefix_offset}.self_attn.q_proj.bias -> {prefix}.self_attn.q_proj.bias",
                        f"{prefix_offset}.self_attn.k_proj.bias -> {prefix}.self_attn.k_proj.bias",
                        f"{prefix_offset}.self_attn.v_proj.bias -> {prefix}.self_attn.v_proj.bias",
                    ]
                    if use_gated_attn_layer:
                        aoa_statements.append(
                            f"{prefix_offset}.self_attn.gate_proj.bias -> {prefix}.self_attn.gate_proj.bias"
                        )
            else:
                fleet_key = f"{prefix_offset}.self_attn.qkv_proj.weight"
                fused_tmp = f"{prefix}.self_attn.qkv_fused_tmp"

                # Step 1: Transpose fleet weight
                aoa_statements.append(f"{fleet_key}^T -> {fused_tmp}")

                heads_per_group = num_heads // num_kv_heads

                # Step 2: Split into per-group chunks along axis=0
                chunk_names = []
                bias_chunk_names = []
                for g in range(num_kv_heads):
                    for h in range(heads_per_group):
                        for c in range(head_dim_chunks):
                            chunk_names.append(f"{prefix}.self_attn._q_g{g}_h{h}_c{c}")
                            bias_chunk_names.append(f"{prefix}.self_attn._q_g{g}_h{h}_c{c}_bias")
                    if use_gated_attn_layer:
                        for h in range(heads_per_group):
                            for c in range(v_head_dim_chunks):
                                chunk_names.append(f"{prefix}.self_attn._gate_g{g}_h{h}_c{c}")
                                bias_chunk_names.append(f"{prefix}.self_attn._gate_g{g}_h{h}_c{c}_bias")
                    for c in range(head_dim_chunks):
                        chunk_names.append(f"{prefix}.self_attn._k_g{g}_c{c}")
                        bias_chunk_names.append(f"{prefix}.self_attn._k_g{g}_c{c}_bias")
                    for c in range(v_head_dim_chunks):
                        chunk_names.append(f"{prefix}.self_attn._v_g{g}_c{c}")
                        bias_chunk_names.append(f"{prefix}.self_attn._v_g{g}_c{c}_bias")
                aoa_statements.append(f"{fused_tmp} -> {','.join(chunk_names)}, axis=0")

                # Step 3: Reassemble q_proj and gate_proj separately
                q_ordered = []
                gate_ordered = []
                bias_q_ordered = []
                bias_gate_ordered = []
                for g in range(num_kv_heads):
                    for h in range(heads_per_group):
                        for c in range(head_dim_chunks):
                            q_ordered.append(f"{prefix}.self_attn._q_g{g}_h{h}_c{c}")
                            bias_q_ordered.append(f"{prefix}.self_attn._q_g{g}_h{h}_c{c}_bias")
                    if use_gated_attn_layer:
                        for h in range(heads_per_group):
                            for c in range(v_head_dim_chunks):
                                gate_ordered.append(f"{prefix}.self_attn._gate_g{g}_h{h}_c{c}")
                                bias_gate_ordered.append(f"{prefix}.self_attn._gate_g{g}_h{h}_c{c}_bias")
                aoa_statements.append(f"{','.join(q_ordered)} -> {prefix}.self_attn.q_proj.weight, axis=0")
                if use_gated_attn_layer:
                    aoa_statements.append(f"{','.join(gate_ordered)} -> {prefix}.self_attn.gate_proj.weight, axis=0")

                # k_proj
                k_ordered = []
                bias_k_ordered = []
                for g in range(num_kv_heads):
                    for c in range(head_dim_chunks):
                        k_ordered.append(f"{prefix}.self_attn._k_g{g}_c{c}")
                        bias_k_ordered.append(f"{prefix}.self_attn._k_g{g}_c{c}_bias")
                aoa_statements.append(f"{','.join(k_ordered)} -> {prefix}.self_attn.k_proj.weight, axis=0")

                # v_proj
                v_ordered = []
                bias_v_ordered = []
                for g in range(num_kv_heads):
                    for c in range(v_head_dim_chunks):
                        v_ordered.append(f"{prefix}.self_attn._v_g{g}_c{c}")
                        bias_v_ordered.append(f"{prefix}.self_attn._v_g{g}_c{c}_bias")
                aoa_statements.append(f"{','.join(v_ordered)} -> {prefix}.self_attn.v_proj.weight, axis=0")

                if config.attention_bias:
                    aoa_statements.append(
                        f"{prefix_offset}.self_attn.qkv_proj.bias -> {','.join(bias_chunk_names)}, axis=0"
                    )
                    aoa_statements.append(f"{','.join(bias_q_ordered)} -> {prefix}.self_attn.q_proj.bias, axis=0")
                    if use_gated_attn_layer:
                        aoa_statements.append(
                            f"{','.join(bias_gate_ordered)} -> {prefix}.self_attn.gate_proj.bias, axis=0"
                        )
                    aoa_statements.append(f"{','.join(bias_k_ordered)} -> {prefix}.self_attn.k_proj.bias, axis=0")
                    aoa_statements.append(f"{','.join(bias_v_ordered)} -> {prefix}.self_attn.v_proj.bias, axis=0")

    # MoE gate / expert weight inverse mapping
    moe_layer_end = num_hidden_layers if config.use_dense_mtp else num_hidden_layers + num_nextn_predict_layers
    for layer_idx in range(config.first_k_dense_replace, moe_layer_end):
        layer_idx_offset = layer_idx + num_head_empty_layers
        prefix_offset = f"{model_prefix}layers.{layer_idx_offset}"
        prefix = f"model.layers.{layer_idx}"
        if layer_idx >= num_hidden_layers:
            prefix_offset += ".transformer_layer"

        use_fused_weight = config.moe_expert_fusion
        if (
            getattr(config, "fp8", False)
            and (config.moe_expert_fusion is False)
            and getattr(config, "moe_deep_gemm", False)
        ):
            import logging

            logging.getLogger(__name__).warning(
                "For fp8 deep_gemm, moe_expert_fusion should be True. Skipping fused weight in inv_aoa."
            )
            use_fused_weight = False
        if (
            getattr(config, "fp8", False)
            and config.moe_expert_fusion
            and getattr(config, "moe_deep_gemm", None) is False
        ):
            use_fused_weight = False

        if use_fused_weight:
            ep_weight1 = []
            ep_weight2 = []
            for expert_id in range(config.n_routed_experts):
                ep_weight1.append(f"{prefix}.mlp.experts.{expert_id}.up_gate_proj.weight")
                ep_weight2.append(f"{prefix}.mlp.experts.{expert_id}.down_proj.weight")
            group_gemm1 = ",".join(ep_weight1)
            group_gemm2 = ",".join(ep_weight2)
            aoa_statements += [
                f"{prefix_offset}.mlp.grouped_gemm_experts.weight1 -> {group_gemm1}, axis=0",
                f"{prefix_offset}.mlp.grouped_gemm_experts.weight2 -> {group_gemm2}, axis=0",
            ]
        else:
            if getattr(config, "fd_fallback", False):
                ep_weight1 = []
                ep_weight2 = []
                for expert_id in range(num_experts):
                    ep_weight1.append(f"{prefix_offset}.mlp.experts.{expert_id}.up_gate_proj.weight")
                    ep_weight2.append(f"{prefix_offset}.mlp.experts.{expert_id}.down_proj.weight")
                group1 = ",".join(ep_weight1)
                group2 = ",".join(ep_weight2)
                aoa_statements += [
                    f"{prefix_offset}.mlp.experts.up_gate_proj -> {group1}, axis=0",
                    f"{prefix_offset}.mlp.experts.down_proj -> {group2}, axis=0",
                ]

        if n_shared_experts > 0:
            aoa_statements += [
                f"{prefix_offset}.mlp.shared_experts.down_proj.weight^T -> "
                f"{prefix}.block_sparse_moe.shared_experts.w2.weight",
                f"{prefix_offset}.mlp.shared_experts.up_gate_proj.weight -> "
                f"{prefix_offset}.block_sparse_moe.shared_experts.gate_proj.weight, "
                f"{prefix_offset}.block_sparse_moe.shared_experts.up_proj.weight, fused_ffn",
                f"{prefix_offset}.block_sparse_moe.shared_experts.gate_proj.weight^T -> "
                f"{prefix}.block_sparse_moe.shared_experts.w1.weight",
                f"{prefix_offset}.block_sparse_moe.shared_experts.up_proj.weight^T -> "
                f"{prefix}.block_sparse_moe.shared_experts.w3.weight",
            ]

        aoa_statements += [
            f"{prefix_offset}.mlp.gate.weight -> {prefix}.block_sparse_moe.gate.weight",
            f"{prefix_offset}.mlp.gate.e_score_correction_bias -> {prefix}.block_sparse_moe.e_score_correction_bias",
        ]
        if getattr(config, "routed_scaling_factor_learnable", False):
            aoa_statements += [
                f"{prefix_offset}.mlp.gate.routed_scaling_factor_param -> "
                f"{prefix}.block_sparse_moe.gate.routed_scaling_factor_param",
            ]

        if config.moe_latent_size is not None and config.moe_latent_size > 0:
            aoa_statements += [
                f"{prefix_offset}.mlp.fc1_latent_proj.weight^T -> {prefix}.block_sparse_moe.fc1_latent_proj.weight ",
                f"{prefix_offset}.mlp.fc2_latent_proj.weight^T -> {prefix}.block_sparse_moe.fc2_latent_proj.weight",
            ]

        if using_sonic_moe:
            aoa_statements += [
                f"{prefix_offset}.mlp.experts.{expert_id}.up_gate_proj.weight -> "
                f"{prefix_offset}.block_sparse_moe.experts.{expert_id}.w1.weight, "
                f"{prefix_offset}.block_sparse_moe.experts.{expert_id}.w3.weight, axis=0"
                for expert_id in range(config.n_routed_experts)
            ]
        else:
            aoa_statements += [
                f"{prefix_offset}.mlp.experts.{expert_id}.up_gate_proj.weight -> "
                f"{prefix_offset}.block_sparse_moe.experts.{expert_id}.w1.weight, "
                f"{prefix_offset}.block_sparse_moe.experts.{expert_id}.w3.weight, axis=1"
                for expert_id in range(config.n_routed_experts)
            ]

        if not using_sonic_moe:
            aoa_statements += (
                [
                    f"{prefix_offset}.block_sparse_moe.experts.{expert_id}.w1.weight^T -> "
                    f"{prefix}.block_sparse_moe.experts.{expert_id}.w1.weight"
                    for expert_id in range(config.n_routed_experts)
                ]
                + [
                    f"{prefix_offset}.block_sparse_moe.experts.{expert_id}.w3.weight^T -> "
                    f"{prefix}.block_sparse_moe.experts.{expert_id}.w3.weight"
                    for expert_id in range(config.n_routed_experts)
                ]
                + [
                    f"{prefix_offset}.mlp.experts.{expert_id}.down_proj.weight^T-> "
                    f"{prefix}.block_sparse_moe.experts.{expert_id}.w2.weight"
                    for expert_id in range(config.n_routed_experts)
                ]
            )

    return {"aoa_statements": aoa_statements}


# ===========================================================================
# TargetFleetCapabilities
# ===========================================================================

_MUON_CRITICAL_FIELDS = [
    "q_lora_rank",
    "qk_nope_head_dim",
    "qk_rope_head_dim",
    "kv_lora_rank",
    "v_head_dim",
    "head_dim",
    "multi_latent_attention",
    "use_gated_attn",
    "muon_configs",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "moe_intermediate_size",
    "intermediate_size",
    "n_routed_experts",
    "mtp_num_layers",
    "num_nextn_predict_layers",
    "moe_expert_fusion",
    # Window/SWA
    "sliding_window",
    "window_attn_skip_freq",
    "swa_head_dim",
    "swa_v_head_dim",
    "swa_num_attention_heads",
    "swa_num_key_value_heads",
    "swa_vha_q_lora_rank",
    # VHA
    "use_vha_attention",
    "vha_q_lora_rank",
    # CSA / DSV4
    "experimental_attention_variant",
    "csa_compress_ratios",
    "csa_dense_mode",
    "dsa_index_n_heads",
    "dsa_index_head_dim",
    # Misc
    "num_empty_layers_add_in_head",
    "softmax_type",
    "add_full_attention_sink_bias",
    "add_swa_attention_sink_bias",
    "routed_scaling_factor_learnable",
    "attention_bias",
]


def _sync_muon_config(model, config):
    """Copy Muon-critical fields from original config to model.config."""
    for field in _MUON_CRITICAL_FIELDS:
        val = getattr(config, field, None)
        if val is not None:
            setattr(model.config, field, val)  # always override


class TargetFleetCapabilities:
    """Attach target MoE fleet model capabilities (AOA + Muon)."""

    @classmethod
    def attach(cls, model, config):
        """Attach fleet capabilities (AOA + Muon) to the model."""
        _sync_muon_config(model, config)
        cls.attach_aoa(model, config)
        cls.attach_muon(model, config)
        return model

    @classmethod
    def attach_aoa(cls, model, config):
        """Attach AOA (Any-Order-Attention) config generators to the model."""
        model._gen_aoa_config = lambda cfg=None: _gen_aoa_config(config if cfg is None else cfg)
        model._gen_inv_aoa_config = lambda cfg=None: _gen_inv_aoa_config(config if cfg is None else cfg)
        return model

    @classmethod
    def attach_muon(cls, model, config):
        """Attach Muon optimizer config generators to the model."""
        model.build_muon_param_info_map = lambda: build_muon_param_info_map(model, model.config)
        model._build_muon_slice_config = lambda: _build_muon_slice_config(model, model.config)
        return model


# ===========================================================================
# TargetForCausalLM
# ===========================================================================


class TargetForCausalLM(PretrainedModel):
    """PaddleFleet entry point for target MoE (like Qwen3MoeForCausalLM).

    Uses ``__new__`` to return a FleetGPTModel directly, matching PaddleFormers
    convention.  Provider construction and capability attachment happen here.
    """

    is_fleet = True

    def __new__(cls, config, training_args=None):
        # Hybrid parallel config defaults (PaddleFormers convention).
        config.tensor_model_parallel_size = max(getattr(config, "tensor_model_parallel_size", 1), 1)
        config.pipeline_model_parallel_size = max(getattr(config, "pipeline_model_parallel_size", 1), 1)
        config.virtual_pipeline_model_parallel_size = max(getattr(config, "virtual_pipeline_model_parallel_size", 1), 1)
        config.expert_model_parallel_size = max(getattr(config, "expert_model_parallel_size", 1), 1)
        config.context_parallel_size = max(getattr(config, "context_parallel_size", 1), 1)

        # Build provider (transform_rules handles field-name translation).
        provider = TargetProvider.from_config(config)
        apply_target_config_overrides(provider, config)
        validate_target_provider(provider, config, training_args=training_args)
        if training_args is not None:
            apply_training_args_to_provider(provider, training_args)
            apply_final_precision_overrides(provider, config)

        # Build model via PaddleFleet gpt_builder.
        gpt_model = provider.provide()

        # Attach target-specific capabilities (AOA + Muon).
        TargetFleetCapabilities.attach(gpt_model, config)

        # PaddleFormers convention: attach config and fleet flag.
        gpt_model.config_to_save = config

        return gpt_model

    @classmethod
    def from_config(cls, config, **kwargs):
        """Create a model from config (PaddleFormers convention).

        Passes ``training_args`` through if provided, matching the target
        pattern where runtime args (TP/PP sizes, dtype) are separate from
        model config.
        """
        return cls(config, **kwargs)
