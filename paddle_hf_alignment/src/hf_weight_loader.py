# Copyright (c) 2026. Echo-MTP experiment — new file, no existing code modified.
"""hf_weight_loader.py — Load HF safetensors checkpoints into an target GPTModel.

This is a STANDALONE file that does NOT modify any existing code in this directory.
It implements the HF→PaddleFleet key mapping discovered by comparing
  * the HF index/shard key names  (model.layers.N.block_sparse_moe.* / self_attn.*)
  * the GPTModel state_dict keys  (model.layers.N.mlp.* / self_attn.*)

Key-mapping rules (fully determined from shape inspection):
──────────────────────────────────────────────────────────
EMBED / HEAD
  model.embed_tokens.weight            → model.embedding.embed_tokens.weight
  lm_head.weight                       → model.lm_head.weight
  model.norm.weight                    → model.final_layernorm.weight  (if present)

SELF-ATTN (all projections are stored transposed in HF relative to model)
  self_attn.q_a_proj.weight    [512,1024]  → self_attn.q_a_proj.weight    [1024,512]  T
  self_attn.q_b_proj.weight    [4096,512]  → self_attn.q_b_proj.weight    [512,4096]  T
  self_attn.kv_a_proj_with_mqa.weight  [576,1024] → [1024,576]  T
  self_attn.kv_b_proj.weight   [7168,512]  → self_attn.kv_b_proj.weight   [512,7168]  T
  self_attn.o_proj.weight      [1024,4096] → self_attn.o_proj.weight      [4096,1024] T
  self_attn.q_a_layernorm.weight  [512]    → direct (model has same key)
  self_attn.kv_a_layernorm.weight [512]    → direct
  self_attn.gate_proj.weight              → SKIP (not in model state_dict)

MOE (per-layer block_sparse_moe.*)
  gate.weight                            → mlp.gate.weight               direct
  e_score_correction_bias                → mlp.gate.e_score_correction_bias  direct
  experts.N.w1.weight  [1024,512]  \
                                     concat axis=0 → mlp.experts.N.up_gate_proj.weight [1024,2048]
  experts.N.w3.weight  [1024,512]  /
  experts.N.w2.weight  [512,1024]        → mlp.experts.N.down_proj.weight  [1024,1024]  T
  shared_experts.w1.weight [1024,1024] \
                                     concat axis=0 → mlp.shared_experts.up_gate_proj.weight [1024,2048]
  shared_experts.w3.weight [1024,1024] /
  shared_experts.w2.weight [1024,1024]   → mlp.shared_experts.down_proj.weight [1024,1024]  direct
  fc1_latent_proj.weight                 → SKIP (not in model state_dict)
  fc2_latent_proj.weight                 → SKIP

MTP LAYER (backbone layers + 1 MTP head — same key structure except MTP-specific)
  layers.24.enorm.weight     → direct
  layers.24.hnorm.weight     → direct
  layers.24.eh_proj.weight   → direct
  layers.24.transformer_layer.* → SKIP (alias to layer 23 via mtp_reuse_last_layer)
  layers.24.norm.weight      → direct (MTP final norm if present)

MISC
  model.layers.N.input_layernorm.weight         → direct
  model.layers.N.post_attention_layernorm.weight → direct
  model.layers.N.mlp.gate.routed_scaling_factor_param → direct (stays in mlp.gate.*)

Usage:
  from hf_weight_loader import build_target_model, load_hf_safetensors_to_model
  model = build_target_model(cfg_dict)
  stats = load_hf_safetensors_to_model(model, ckpt_dir)
"""
from __future__ import annotations

import json
import os
import sys
import logging
from pathlib import Path
from typing import Any

import paddle
import numpy as np

logger = logging.getLogger("hf_weight_loader")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO,
                        format="[%(asctime)s][%(levelname)s] %(message)s")

# ---------------------------------------------------------------------------
# Path helpers — inject packaged PaddleFleet + eb5 code without pip install
# ---------------------------------------------------------------------------
_HERE = Path(__file__).parent          # delivery/src/
_DELIVERY_ROOT = _HERE.parent
_RUNTIME_ROOT = _DELIVERY_ROOT / "runtime"


def _ensure_paths() -> None:
    runtime_paths = [
        _RUNTIME_ROOT / "PaddleFleet" / "src",
        _RUNTIME_ROOT,
        _HERE,
    ]
    missing = [str(p) for p in runtime_paths if not p.is_dir()]
    if missing:
        raise RuntimeError(f"Incomplete delivery; missing runtime directories: {missing}")
    for path in runtime_paths:
        p = str(path)
        if p not in sys.path:
            sys.path.insert(0, p)


# ---------------------------------------------------------------------------
# Build model from a config dict (bypasses AutoConfig / paddleformers)
# ---------------------------------------------------------------------------
def build_target_model(cfg: dict, *, lazy: bool = True, dtype: str = "float32") -> Any:
    """Build an target GPTModel from a raw config dict.

    Args:
        cfg:   contents of config.json (HF-style the target format).
        lazy:  if True (default), use paddle.LazyGuard() so weights are not
               materialised on GPU yet (useful for loading from ckpt).
        dtype: 'float32' or 'bfloat16'. Model parameters are initialised in
               this dtype. Must match the dtype used in load_hf_safetensors_to_model.

    Returns:
        The GPTModel (PaddleFleet Megatron-style).
    """
    _ensure_paths()

    from fleet_model.target.modeling import TargetProvider  # noqa: PLC0415

    # ── config fields that map 1-1 to provider dataclass fields ──────────
    _DIRECT = [
        "hidden_size", "intermediate_size", "num_attention_heads",
        "num_key_value_heads", "head_dim",
        "n_routed_experts", "num_experts_per_tok", "n_shared_experts",
        "moe_intermediate_size",
        "vocab_size", "q_lora_rank", "kv_lora_rank",
        "qk_nope_head_dim", "qk_rope_head_dim", "v_head_dim",
        "rms_norm_eps", "rope_theta", "use_qk_norm",
        "num_nextn_predict_layers", "mtp_reuse_last_layer",
        "tie_word_embeddings", "use_gated_attn", "hidden_act",
        "rope_type", "scoring_func", "topk_method", "n_group", "topk_group",
        "routed_scaling_factor", "routed_scaling_factor_learnable",
        "use_routing_bias", "moe_router_load_balancing_type",
    ]
    p = TargetProvider()
    for k in _DIRECT:
        if k in cfg:
            setattr(p, k, cfg[k])

    # ── num_hidden_layers ───────────────────────────────────────────────
    p.num_hidden_layers = cfg["num_hidden_layers"]
    p.num_layers = cfg["num_hidden_layers"]

    # ── MoE shared expert count (config uses n_shared_experts, not num_shared_experts) ─
    if "n_shared_experts" in cfg:
        p.n_shared_experts = cfg["n_shared_experts"]
    elif "num_shared_experts" in cfg:
        p.n_shared_experts = cfg["num_shared_experts"]

    # ── MoE intermediate size (required; provider field is moe_intermediate_size) ─
    if "moe_intermediate_size" in cfg:
        p.moe_intermediate_size = cfg["moe_intermediate_size"]
    elif "moe_ffn_dim" in cfg:
        p.moe_intermediate_size = cfg["moe_ffn_dim"]
    else:
        p.moe_intermediate_size = cfg.get("intermediate_size", 1024)

    # ── MoE latent size (controls expert fc1_latent_proj / fc2_latent_proj) ──
    if hasattr(p, "moe_latent_size") and "moe_latent_size" in cfg:
        p.moe_latent_size = cfg["moe_latent_size"]

    # ── MLA / multi-latent attention (must be True to get q_a_proj, kv_b_proj etc.) ─
    p.multi_latent_attention = bool(cfg.get("multi_latent_attention", True))

    # ── QK norm and attention/router compatibility aliases ───────────────
    p.use_qk_norm = bool(cfg.get("use_qk_norm", True))
    p.gated_attention = bool(cfg.get("use_gated_attn", False))
    # Sigmoid routing in the training config normalizes selected top-k scores
    # before applying either fixed or per-expert routed scaling.
    p.norm_topk_prob = bool(cfg.get("norm_topk_prob", cfg.get("scoring_func") == "sigmoid"))
    if isinstance(p.hidden_act, str):
        p.hidden_act = getattr(paddle.nn.functional, p.hidden_act)
    p.rotary_base = cfg.get("rope_theta", p.rotary_base)
    if cfg.get("rope_scaling") is None:
        p.rope_type = cfg.get("rope_type", "rope")
        p.rotary_scaling_factor = 1.0
    nnp = cfg.get("num_nextn_predict_layers", 0)
    p.mtp_num_layers = nnp
    p.mtp_enabled = bool(cfg.get("use_mtp", False)) and nnp > 0

    # ── dtype ────────────────────────────────────────────────────────────
    # params_dtype controls the dtype of model parameters (independent of bf16 flag).
    # Must set both bf16=True AND params_dtype=bfloat16 to get bfloat16 params.
    if dtype == "bfloat16":
        p.bf16 = True
        p.params_dtype = paddle.bfloat16
    else:
        p.bf16 = False
        p.params_dtype = paddle.float32

    # ── rope_scaling: must be None or dict, NOT float ───────────────────
    p.rope_scaling = None
    # Disable rope fusion: it doesn't support dynamic/non-pipelined inference
    p.apply_rope_fusion = False

    # ── INFERENCE FIX: disable moe_router_force_load_balancing ──────────
    # TargetProvider defaults this to True (training-time load-balance trick).
    # At inference it replaces gate logits with pure randn() via RandomSTE,
    # making every forward nondeterministic: PPL≈1e9 on any tokenizer, repeat
    # tokens, self-inconsistency. Must be False for correct inference.
    p.moe_router_force_load_balancing = False

    logger.info("Building target model (layers=%d, experts=%d, mtp=%d) …",
                p.num_hidden_layers, p.n_routed_experts, nnp)

    # Ensure GPU is initialized (LazyGuard requires a device context)
    paddle.set_device("gpu")

    # Set default dtype so that nn.Linear and other modules that don't read
    # params_dtype (e.g. MoE fc_latent_proj) also use the correct dtype.
    paddle.set_default_dtype(dtype)

    if lazy:
        with paddle.LazyGuard():
            model = p.provide()
    else:
        model = p.provide()

    n_keys = len(model.state_dict())
    logger.info("Model built: %d state_dict keys", n_keys)
    return model


# ---------------------------------------------------------------------------
# HF key → GPTModel key mapping helpers
# ---------------------------------------------------------------------------

def _map_hf_key(hf_key: str) -> str | None:
    """Return the corresponding GPTModel state_dict key, or None to skip.

    Rules are applied in priority order; returns None means "skip this tensor".
    """
    k = hf_key

    # ── top-level embed / head ───────────────────────────────────────────
    if k == "model.embed_tokens.weight":
        return "model.embedding.embed_tokens.weight"
    if k == "lm_head.weight":
        return "model.lm_head.weight"
    if k == "model.norm.weight":
        return "model.norm.weight"  # same key in paddle GPTModel state_dict

    # ── skip transformer_layer aliases (mtp_reuse_last_layer) ───────────
    if ".transformer_layer." in k:
        return None

    # ── per-layer prefix: model.layers.N.* ──────────────────────────────
    if not k.startswith("model.layers."):
        return None  # unknown top-level — skip

    # strip "model.layers.N." prefix, keep suffix
    parts = k.split(".", 3)  # ["model", "layers", "N", rest]
    if len(parts) < 4:
        return None
    layer_prefix = ".".join(parts[:3])  # "model.layers.N"
    layer_idx = int(parts[2])
    suffix = parts[3]                   # everything after "model.layers.N."

    # ── MTP-specific top-level subkeys (enorm, hnorm, eh_proj, norm) ────
    # These live directly under model.layers.24.* regardless of mtp_reuse_last_layer.
    if suffix in ("enorm.weight", "hnorm.weight", "norm.weight"):
        return f"{layer_prefix}.{suffix}"
    # MTP head final norm: HF names it shared_head.norm.weight, model uses norm.weight
    if suffix == "shared_head.norm.weight":
        return f"{layer_prefix}.norm.weight"
    if suffix == "eh_proj.weight":
        # HF stores [1024, 2048] (out, in); model expects [2048, 1024] (in, out)
        return f"__transpose_latent__{layer_prefix}.{suffix}"

    # ── MTP layer (layer_idx >= 24): backbone keys live under transformer_layer ──
    # When mtp_reuse_last_layer=True, layer 24 stores backbone weights as aliases
    # to layer 23 under model.layers.24.transformer_layer.*.
    # There are NO model.layers.24.self_attn.* or model.layers.24.mlp.* keys.
    # All backbone-style suffixes for this layer must be re-routed through transformer_layer.
    _MTP_LAYER_START = 24  # num_hidden_layers for this model
    if layer_idx >= _MTP_LAYER_START:
        # redirect layer_prefix for all backbone keys
        layer_prefix = f"{layer_prefix}.transformer_layer"
        # fall through to standard backbone suffix handling below

    # ── layernorms ───────────────────────────────────────────────────────
    if suffix in ("input_layernorm.weight", "post_attention_layernorm.weight"):
        return f"{layer_prefix}.{suffix}"

    # ── self_attn ────────────────────────────────────────────────────────
    if suffix.startswith("self_attn."):
        attn_sub = suffix[len("self_attn."):]
        if attn_sub == "gate_proj.weight":
            return f"{layer_prefix}.self_attn.gate_proj.weight"
        # direct (layernorm weights, shape already matches)
        if attn_sub in ("q_a_layernorm.weight", "kv_a_layernorm.weight"):
            return f"{layer_prefix}.self_attn.{attn_sub}"
        # projection weights — all need transpose (handled in loader)
        if attn_sub in ("q_a_proj.weight", "q_b_proj.weight",
                         "kv_a_proj_with_mqa.weight", "kv_b_proj.weight",
                         "o_proj.weight"):
            return f"{layer_prefix}.self_attn.{attn_sub}"
        return None  # unknown attn sub-key

    # ── mlp.gate.routed_scaling_factor_param ────────────────────────────
    if suffix == "mlp.gate.routed_scaling_factor_param":
        return f"{layer_prefix}.mlp.gate.routed_scaling_factor_param"

    # ── block_sparse_moe.* ───────────────────────────────────────────────
    if suffix.startswith("block_sparse_moe."):
        moe_sub = suffix[len("block_sparse_moe."):]

        if moe_sub == "gate.weight":
            return f"{layer_prefix}.mlp.gate.weight"
        # Exports from 2026-06-30 onward nest the learnable scale under the gate instead of
        # emitting `mlp.gate.routed_scaling_factor_param`. Unmapped, it silently stays at its
        # init (config.routed_scaling_factor = 2.5) while the real values are 0.1-0.3.
        if moe_sub == "gate.routed_scaling_factor_param":
            return f"{layer_prefix}.mlp.gate.routed_scaling_factor_param"
        if moe_sub == "e_score_correction_bias":
            return f"{layer_prefix}.mlp.gate.e_score_correction_bias"
        if moe_sub in ("fc1_latent_proj.weight", "fc2_latent_proj.weight"):
            # Map to model mlp.fc1_latent_proj / fc2_latent_proj (need transpose)
            sub_name = moe_sub  # "fc1_latent_proj.weight" or "fc2_latent_proj.weight"
            return f"__transpose_latent__{layer_prefix}.mlp.{sub_name}"

        # shared_experts
        if moe_sub.startswith("shared_experts."):
            sh_sub = moe_sub[len("shared_experts."):]
            if sh_sub in ("w1.weight", "w3.weight"):
                # will be fused in loader
                return f"__fuse_shared_up_gate__{layer_prefix}"
            if sh_sub == "w2.weight":
                return f"{layer_prefix}.mlp.shared_experts.down_proj.weight"
            return None

        # routed experts
        if moe_sub.startswith("experts."):
            rest = moe_sub[len("experts."):]
            # rest = "N.w1.weight" / "N.w2.weight" / "N.w3.weight"
            parts2 = rest.split(".", 1)
            if len(parts2) != 2:
                return None
            eidx, sub = parts2
            if sub in ("w1.weight", "w3.weight"):
                return f"__fuse_expert_up_gate__{layer_prefix}__experts.{eidx}"
            if sub == "w2.weight":
                return f"{layer_prefix}.mlp.experts.{eidx}.down_proj.weight"
            return None

        return None  # unknown moe sub-key

    return None  # unrecognised suffix


# Projection keys that need to be transposed (HF stores [out,in], model uses [in,out])
_TRANSPOSE_SUFFIXES = frozenset([
    "self_attn.q_a_proj.weight",
    "self_attn.q_b_proj.weight",
    "self_attn.kv_a_proj_with_mqa.weight",
    "self_attn.kv_b_proj.weight",
    "self_attn.o_proj.weight",
    "self_attn.gate_proj.weight",
])


def _needs_transpose(hf_key: str) -> bool:
    for sfx in _TRANSPOSE_SUFFIXES:
        if hf_key.endswith(sfx):
            return True
    return False


# ---------------------------------------------------------------------------
# Main loader
# ---------------------------------------------------------------------------
def load_hf_safetensors_to_model(
    model: Any,
    ckpt_dir: str,
    *,
    index_file: str = "model.safetensors.index.json",
    dtype: str = "float32",
    verbose: bool = False,
) -> dict:
    """Load all HF safetensors shards into *model* in-place.

    Args:
        model:       GPTModel returned by build_target_model().
        ckpt_dir:    Directory containing the *.safetensors shards and index.
        index_file:  Name of the shard-index JSON (default: standard HF name).
        dtype:       Target dtype string for paddle tensors ('float32'/'bfloat16').
        verbose:     Log every key mapping (very noisy for 20B models).

    Returns:
        dict with stats: loaded, skipped, missing, unexpected keys.
    """
    try:
        from safetensors import safe_open  # noqa: PLC0415
    except ImportError as e:
        raise ImportError(
            "safetensors is required. Install with: pip install safetensors"
        ) from e

    idx_path = os.path.join(ckpt_dir, index_file)
    if not os.path.isfile(idx_path):
        raise FileNotFoundError(f"Shard index not found: {idx_path}")

    with open(idx_path) as f:
        idx = json.load(f)
    weight_map: dict[str, str] = idx["weight_map"]

    # Collect model state_dict keys for cross-checking
    model_sd = model.state_dict()
    model_keys = set(model_sd.keys())

    # Build shard → list[hf_key] mapping so each shard is opened once
    shard_to_keys: dict[str, list[str]] = {}
    for hf_key, shard_file in weight_map.items():
        shard_to_keys.setdefault(shard_file, []).append(hf_key)

    # Accumulate tensors to set at the end (expert fusion requires buffering)
    updates: dict[str, Any] = {}   # model_key → paddle Tensor
    # For fused expert up_gate: {(layer_prefix, eidx): {"w1": ..., "w3": ...}}
    expert_buffers: dict[tuple[str, str], dict[str, Any]] = {}
    # For fused shared up_gate: {layer_prefix: {"w1": ..., "w3": ...}}
    shared_buffers: dict[str, dict[str, Any]] = {}

    stats = {"loaded": 0, "skipped": 0, "fused": 0, "errors": []}
    shards = sorted(shard_to_keys.keys())
    logger.info("Loading %d shards from %s …", len(shards), ckpt_dir)

    for i, shard_file in enumerate(shards):
        shard_path = os.path.join(ckpt_dir, shard_file)
        if not os.path.isfile(shard_path):
            raise FileNotFoundError(f"Shard missing: {shard_path}")

        if (i + 1) % 20 == 0 or i == 0:
            logger.info("  shard %d/%d: %s", i + 1, len(shards), shard_file)

        with safe_open(shard_path, framework="pt") as sf:
            for hf_key in shard_to_keys[shard_file]:
                mapped = _map_hf_key(hf_key)

                # ── skip ──────────────────────────────────────────────
                if mapped is None:
                    stats["skipped"] += 1
                    if verbose:
                        logger.debug("SKIP  %s", hf_key)
                    continue

                # Load tensor as numpy (bfloat16 → float32 safe path)
                t_pt = sf.get_tensor(hf_key)
                arr = t_pt.float().numpy()  # float32 numpy

                # ── latent proj (fc1/fc2) — needs transpose ───────────
                if mapped.startswith("__transpose_latent__"):
                    model_key = mapped[len("__transpose_latent__"):]
                    if model_key in model_keys:
                        updates[model_key] = paddle.to_tensor(arr.T, dtype=dtype)
                        stats["loaded"] += 1
                    else:
                        stats["skipped"] += 1  # not in model (moe_latent_size not set)
                    continue

                # ── fuse up/gate for routed experts ──────────────────
                if mapped.startswith("__fuse_expert_up_gate__"):
                    # encoded as "__fuse_expert_up_gate__{layer_prefix}__experts.{eidx}"
                    rest = mapped[len("__fuse_expert_up_gate__"):]
                    layer_pfx, exp_tag = rest.split("__", 1)
                    # exp_tag = "experts.N"
                    eidx = exp_tag.split(".", 1)[1]  # "N"
                    buf_key = (layer_pfx, eidx)
                    if buf_key not in expert_buffers:
                        expert_buffers[buf_key] = {}
                    hf_sub = hf_key.split(".")[-2]  # "w1" or "w3"
                    expert_buffers[buf_key][hf_sub] = arr

                    if "w1" in expert_buffers[buf_key] and "w3" in expert_buffers[buf_key]:
                        buf = expert_buffers.pop(buf_key)
                        # Routed expert: HF w1/w3 are [out,in] -> transpose to [in,out]
                        # then concat on axis=1 to get up_gate_proj [latent_in, 2*moe_ffn]
                        w1_T = buf["w1"].T  # [in=latent, out=moe_ffn]
                        w3_T = buf["w3"].T  # [in=latent, out=moe_ffm]
                        fused = np.concatenate([w1_T, w3_T], axis=1)  # [latent_in, 2*moe_ffn]
                        model_key = f"{layer_pfx}.mlp.experts.{eidx}.up_gate_proj.weight"
                        updates[model_key] = paddle.to_tensor(fused, dtype=dtype)
                        stats["fused"] += 1
                    continue

                # ── fuse up/gate for shared experts ──────────────────
                if mapped.startswith("__fuse_shared_up_gate__"):
                    layer_pfx = mapped[len("__fuse_shared_up_gate__"):]
                    if layer_pfx not in shared_buffers:
                        shared_buffers[layer_pfx] = {}
                    hf_sub = hf_key.split(".")[-2]  # "w1" or "w3"
                    shared_buffers[layer_pfx][hf_sub] = arr

                    if "w1" in shared_buffers[layer_pfx] and "w3" in shared_buffers[layer_pfx]:
                        buf = shared_buffers.pop(layer_pfx)
                        # Shared expert: HF w1/w3 [out,in] -> T -> concat axis=1
                        w1_T = buf["w1"].T  # [hidden, moe_ffn]
                        w3_T = buf["w3"].T
                        fused = np.concatenate([w1_T, w3_T], axis=1)  # [hidden, 2*moe_ffn]
                        model_key = f"{layer_pfx}.mlp.shared_experts.up_gate_proj.weight"
                        updates[model_key] = paddle.to_tensor(fused, dtype=dtype)
                        stats["fused"] += 1
                    continue

                # ── transpose if needed ───────────────────────────────
                if _needs_transpose(hf_key):
                    arr = arr.T

                # ── w2 (shared expert down_proj) needs transpose ──────
                # shared w2 [1024,1024] (HF [out,in]) -> T = [1024,1024] (same shape, but different semantics)
                # Actually shared w2 maps moe_ffn=1024 -> hidden=1024, same shape T=same
                # But to be consistent: T is correct for W=[out,in] -> [in,out]
                if hf_key.endswith(".w2.weight") and "shared_experts" in hf_key:
                    arr = arr.T  # [1024,1024] -> [1024,1024] (T of symmetric — still correct)
                if hf_key.endswith(".w2.weight") and "experts." in hf_key and "shared" not in hf_key:
                    arr = arr.T  # [512,1024] → [1024,512] (moe_ffm, latent_out)

                # ── store ─────────────────────────────────────────────
                if mapped in model_keys:
                    updates[mapped] = paddle.to_tensor(arr, dtype=dtype)
                    stats["loaded"] += 1
                    if verbose:
                        logger.debug("LOAD  %s → %s %s", hf_key, mapped, list(arr.shape))
                else:
                    stats["errors"].append(f"model key not found: {mapped} (from {hf_key})")
                    if verbose:
                        logger.warning("MISS  %s → %s (not in model sd)", hf_key, mapped)

    # ── set all at once ───────────────────────────────────────────────────
    logger.info("Applying %d updates to model state_dict …", len(updates))
    # Validate shapes before applying
    shape_errors = []
    for mkey, tensor in updates.items():
        if mkey in model_keys:
            expected = list(model_sd[mkey].shape)
            got = list(tensor.shape)
            if expected != got:
                shape_errors.append(f"{mkey}: expected {expected}, got {got}")

    if shape_errors:
        logger.warning("Shape mismatches (%d):", len(shape_errors))
        for e in shape_errors[:20]:
            logger.warning("  %s", e)

    # Cast each update tensor to match the model param's actual dtype.
    # Some params (e.g. MoE gate, fc_latent_proj) are forced to float32 by
    # the model code even when the global dtype is bfloat16.
    for mkey in list(updates.keys()):
        if mkey in model_keys:
            target_dtype = model_sd[mkey].dtype
            if updates[mkey].dtype != target_dtype:
                updates[mkey] = updates[mkey].cast(target_dtype)

    # Snapshot which model keys we are about to fill (set_state_dict consumes
    # the dict, so this must be captured before the call).
    loaded_model_keys = set(updates.keys())

    model.set_state_dict(updates)

    # Report missing model keys.
    # NOTE: snapshot the update keys BEFORE set_state_dict — paddle's
    # set_state_dict mutates/consumes the passed dict, so reading
    # updates.keys() afterwards under-counts (only leftover alias keys remain).
    missing = model_keys - loaded_model_keys
    # Filter out transformer_layer aliases (expected missing)
    real_missing = [k for k in missing if "transformer_layer" not in k]
    stats["missing_model_keys"] = len(real_missing)
    stats["shape_errors"] = len(shape_errors)

    logger.info(
        "Done. loaded=%d fused=%d skipped=%d missing_model_keys=%d shape_errors=%d",
        stats["loaded"], stats["fused"], stats["skipped"],
        stats["missing_model_keys"], stats["shape_errors"],
    )
    if real_missing:
        logger.warning("First 10 missing model keys: %s", real_missing[:10])
    return stats


# ---------------------------------------------------------------------------
# Convenience: build + load in one call
# ---------------------------------------------------------------------------
def build_and_load(ckpt_dir: str, *, dtype: str = "float32", verbose: bool = False) -> tuple[Any, dict]:
    """Build model from ckpt_dir/config.json, then load weights.

    Returns (model, load_stats).
    """
    cfg_path = os.path.join(ckpt_dir, "config.json")
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"config.json not found in {ckpt_dir}")
    with open(cfg_path) as f:
        cfg = json.load(f)

    model = build_target_model(cfg, lazy=True)
    stats = load_hf_safetensors_to_model(model, ckpt_dir, dtype=dtype, verbose=verbose)
    return model, stats
