"""torch_weight_loader.py — Load the target V2 HF safetensors into PyTorch model.

Key mapping rules (derived from checkpoint key inspection):
  1. embed/head/norm: direct 1-to-1
  2. self_attn projections: stored [out,in] in HF, nn.Linear expects [out,in] — NO transpose needed
     (PyTorch nn.Linear weight = [out_features, in_features], same as HF convention)
  3. MoE experts w1/w3: stored [out,in] separately → loaded into up_gate_proj [out_features=2*moe_ffn, in_features=latent]
     actually stored as [moe_ffn, latent] each → concat → [2*moe_ffn, latent] for up_gate_proj
  4. MoE experts w2: [latent, moe_ffn] → down_proj [latent, moe_ffn] = direct
  5. shared experts: same pattern as routed experts but shape [moe_ffn, hidden]
  6. eh_proj: stored [hidden, 2*hidden] → nn.Linear weight [hidden, 2*hidden] = direct
  7. layers.24.transformer_layer.*: skip when mtp_reuse_last_layer=True (shared pointer, no separate weights)
                                     load as independent layer when mtp_reuse_last_layer=False
  8. e_score_correction_bias: load as buffer (may or may not be in state_dict)
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch

logger = logging.getLogger("torch_weight_loader")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

_HERE = Path(__file__).parent


def load_config(ckpt_dir: str) -> dict:
    cfg_path = os.path.join(ckpt_dir, "config.json")
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"config.json not found: {cfg_path}")
    with open(cfg_path) as f:
        return json.load(f)


def read_weight_map(ckpt_dir: str) -> Dict[str, str]:
    idx_path = os.path.join(ckpt_dir, "model.safetensors.index.json")
    if not os.path.isfile(idx_path):
        raise FileNotFoundError(idx_path)
    with open(idx_path) as f:
        return json.load(f)["weight_map"]


def inspect_mtp_layout(ckpt_dir: str, num_hidden_layers: int = 24) -> dict:
    """Report what an export ACTUALLY contains, independent of training flags.

    Every export seen so far (both key generations, Echo and DeepSeek-style, shared and
    unshared) materialises each MTP depth as a COMPLETE flat decoder layer under
    ``model.layers.{num_hidden_layers + k}.*``. Inference therefore never needs to know
    whether training aliased those parameters: loading depth k's own keys is correct in
    every case. The only things that must be honoured are (a) how many depths exist and
    (b) that each depth carries its own router correction bias (a buffer, which no
    aliasing path covers — see docs/FIXES.md §7).

    Returned dict keys:
      num_mtp_layers   how many MTP depths the checkpoint stores
      key_style        "flat" (newer) or "transformer_layer" (2026-06-10 export)
      shared_with_prev per depth>0: True if its body is bit-identical to depth 0
      shared_with_backbone_last  per depth: True if body is bit-identical to layer N-1
      bias_max_abs_diff_vs_depth0  per depth>0
    """
    from safetensors import safe_open

    weight_map = read_weight_map(ckpt_dir)
    mtp_layers = sorted(
        L for L in range(num_hidden_layers, num_hidden_layers + 64)
        if f"model.layers.{L}.eh_proj.weight" in weight_map
    )
    style = "transformer_layer" if any(
        k.startswith(f"model.layers.{num_hidden_layers}.transformer_layer.")
        and not k.endswith("routed_scaling_factor_param")
        for k in weight_map
    ) else "flat"

    def get(key):
        with safe_open(os.path.join(ckpt_dir, weight_map[key]), framework="pt") as sf:
            return sf.get_tensor(key)

    probes = [
        "self_attn.q_a_proj.weight",
        "block_sparse_moe.gate.weight",
        "input_layernorm.weight",
        "eh_proj.weight",
    ]
    bias_suffix = "block_sparse_moe.e_score_correction_bias"
    last = num_hidden_layers - 1
    info = {
        "num_mtp_layers": len(mtp_layers),
        "key_style": style,
        "shared_with_prev": {},
        "shared_with_backbone_last": {},
        "bias_max_abs_diff_vs_depth0": {},
    }
    if not mtp_layers:
        return info

    for i, L in enumerate(mtp_layers):
        eq_bb = all(
            torch.equal(get(f"model.layers.{last}.{s}"), get(f"model.layers.{L}.{s}"))
            for s in probes if f"model.layers.{last}.{s}" in weight_map
        )
        info["shared_with_backbone_last"][i] = eq_bb
        if i:
            d0 = mtp_layers[0]
            info["shared_with_prev"][i] = all(
                torch.equal(get(f"model.layers.{d0}.{s}"), get(f"model.layers.{L}.{s}"))
                for s in probes if f"model.layers.{d0}.{s}" in weight_map
            )
            k0, kk = f"model.layers.{d0}.{bias_suffix}", f"model.layers.{L}.{bias_suffix}"
            if k0 in weight_map and kk in weight_map:
                info["bias_max_abs_diff_vs_depth0"][i] = float(
                    (get(k0).float() - get(kk).float()).abs().max()
                )
    return info


def build_torch_model(
    ckpt_dir: str,
    device: str = "cpu",
    dtype: torch.dtype = torch.bfloat16,
    no_mtp: bool = False,
    preserve_mtp_sharing: bool = False,
):
    """Build PyTorch TargetForCausalLM from checkpoint directory.

    The MTP layout is taken from the CHECKPOINT, not from the config's training flags:
    `num_nextn_predict_layers` is set to the number of MTP layers actually exported, and
    `mtp_reuse_last_layer` is normally forced to False so every depth gets its own module
    and is loaded from its own keys. `preserve_mtp_sharing=True` is the explicit Echo
    inference mode: learned decoder Parameters are aliased to the final backbone layer,
    while each MTP module keeps its own non-Parameter buffers such as the routing
    correction bias.

    `no_mtp=True` builds the backbone alone (no MTP modules at all). Target-only evals never
    call the MTP heads anyway, so this is a control: it proves a main-model number cannot be
    influenced by MTP construction or loading.
    """
    # Add src to path so modeling is importable
    src_dir = str(_HERE.parent)
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    from modeling import TargetConfig, TargetForCausalLM  # noqa

    cfg_dict = load_config(ckpt_dir)
    config = TargetConfig(**{
        k: v for k, v in cfg_dict.items()
        if k in TargetConfig.__init__.__code__.co_varnames
    })

    layout = inspect_mtp_layout(ckpt_dir, config.num_hidden_layers)
    n_mtp = layout["num_mtp_layers"]
    if no_mtp:
        logger.info("no_mtp=True -> building backbone only (checkpoint has %d MTP layers)", n_mtp)
        config.num_nextn_predict_layers = 0
        config.use_mtp = False
        config.mtp_reuse_last_layer = False
    elif n_mtp and preserve_mtp_sharing:
        if not config.mtp_reuse_last_layer:
            raise ValueError(
                "preserve_mtp_sharing=True requires checkpoint "
                "mtp_reuse_last_layer=True"
            )
        config.num_nextn_predict_layers = n_mtp
        logger.info(
            "MTP layout from checkpoint: depths=%d, preserving Echo shared "
            "Parameters; correction bias remains depth-specific",
            n_mtp,
        )
    elif n_mtp and (
        n_mtp != config.num_nextn_predict_layers or config.mtp_reuse_last_layer
    ):
        logger.info(
            "MTP layout from checkpoint: depths=%d (config said %d), key_style=%s -> "
            "building independent MTP layers (mtp_reuse_last_layer forced False)",
            n_mtp, config.num_nextn_predict_layers, layout["key_style"],
        )
        config.num_nextn_predict_layers = n_mtp
        config.mtp_reuse_last_layer = False
    logger.info(
        "checkpoint sharing (informational): body==backbone_last %s, body==depth0 %s, "
        "bias max|d| vs depth0 %s",
        layout["shared_with_backbone_last"], layout["shared_with_prev"],
        {k: round(v, 4) for k, v in layout["bias_max_abs_diff_vs_depth0"].items()},
    )

    logger.info("Building TargetForCausalLM (layers=%d, experts=%d, mtp_depths=%d) …",
                config.num_hidden_layers, config.n_routed_experts,
                config.num_nextn_predict_layers)

    with torch.device(device):
        model = TargetForCausalLM(config).to(dtype=dtype)

    # `.to(dtype)` also casts float BUFFERS (the same trap documented for
    # TargetRotaryEmbedding.inv_freq). e_score_correction_bias is fp32 in the
    # checkpoint and is added to sigmoid scores that are O(1e-4..1e-1): bf16
    # resolution near 0.6 is ~0.004, so the cast perturbs the bias by ~0.002 and
    # REORDERS the top-8 boundary. Measured against the fp32 checkpoint values,
    # the bf16 bias flipped ~2.4 of 8 experts for 59% of layer-0 tokens; that was
    # the whole source of the sglang-vs-HF gate-on logits divergence (sglang keeps
    # it fp32 and reproduces top_k(sigmoid(logits)+bias) exactly, 8.000/8).
    # Keep routing-critical buffers in fp32 so the loader writes fp32 values.
    for module in model.modules():
        bias = getattr(module, "e_score_correction_bias", None)
        if isinstance(bias, torch.Tensor) and bias.is_floating_point():
            module.e_score_correction_bias = bias.float()

    return model, config


def _map_layer_suffix(suffix: str, pfx: str) -> Optional[str]:
    """Map a decoder-layer-relative HF suffix onto `pfx`-rooted state_dict key.

    `pfx` is either `model.layers.{i}` (backbone) or
    `model.mtp_heads.{i}.transformer_layer` (independent MTP layer).
    """
    # Layernorms
    if suffix in ("input_layernorm.weight", "post_attention_layernorm.weight"):
        return f"{pfx}.{suffix}"

    # Self-attention
    if suffix.startswith("self_attn."):
        attn_sub = suffix[len("self_attn."):]
        if attn_sub == "gate_proj.weight":
            return f"{pfx}.self_attn.gate_proj.weight"
        if attn_sub in (
            "q_a_proj.weight", "q_b_proj.weight",
            "kv_a_proj_with_mqa.weight", "kv_b_proj.weight", "o_proj.weight",
            "q_a_layernorm.weight", "kv_a_layernorm.weight",
        ):
            return f"{pfx}.self_attn.{attn_sub}"
        return None

    # MoE — routed scaling factor (older exports put it outside block_sparse_moe)
    if suffix == "mlp.gate.routed_scaling_factor_param":
        return f"{pfx}.mlp.gate.routed_scaling_factor_param"

    # MoE — block_sparse_moe
    if suffix.startswith("block_sparse_moe."):
        moe_sub = suffix[len("block_sparse_moe."):]

        if moe_sub == "gate.weight":
            return f"{pfx}.mlp.gate.weight"
        # newer exports nest the learnable scale under the gate
        if moe_sub == "gate.routed_scaling_factor_param":
            return f"{pfx}.mlp.gate.routed_scaling_factor_param"
        if moe_sub == "e_score_correction_bias":
            return f"{pfx}.mlp.gate.e_score_correction_bias"
        if moe_sub in ("fc1_latent_proj.weight", "fc2_latent_proj.weight"):
            return f"{pfx}.mlp.{moe_sub}"
        if moe_sub.startswith("shared_experts."):
            sh_sub = moe_sub[len("shared_experts."):]
            if sh_sub in ("w1.weight", "w3.weight"):
                return f"__fuse_shared__{pfx}__w1w3__{sh_sub}"
            if sh_sub == "w2.weight":
                return f"{pfx}.mlp.shared_experts.down_proj.weight"
            return None
        if moe_sub.startswith("experts."):
            rest = moe_sub[len("experts."):]
            parts2 = rest.split(".", 1)
            if len(parts2) != 2:
                return None
            eidx, sub = parts2
            if sub in ("w1.weight", "w3.weight"):
                return f"__fuse_expert__{pfx}__e{eidx}__w1w3__{sub}"
            if sub == "w2.weight":
                return f"{pfx}.mlp.experts.{eidx}.down_proj.weight"
            return None
        return None

    return None


def _map_hf_to_torch(hf_key: str, mtp_reuse_last_layer: bool, num_hidden_layers: int = 24) -> Optional[str]:
    """Map HF checkpoint key → PyTorch model state_dict key.
    Returns None to skip the key.
    """
    k = hf_key

    # Top-level
    if k == "model.embed_tokens.weight":
        return "model.embed_tokens.weight"
    if k == "lm_head.weight":
        return "lm_head.weight"
    if k == "model.norm.weight":
        return "model.norm.weight"

    if not k.startswith("model.layers."):
        return None

    parts = k.split(".", 3)
    if len(parts) < 4:
        return None
    layer_idx = int(parts[2])
    suffix = parts[3]
    layer_pfx = f"model.layers.{layer_idx}"

    # MTP layer (layer_idx >= num_hidden_layers)
    is_mtp_layer = layer_idx >= num_hidden_layers
    if is_mtp_layer:
        depth = layer_idx - num_hidden_layers
        mtp_pfx = f"model.mtp_heads.{depth}"
        # MTP-specific keys
        if suffix in ("enorm.weight", "hnorm.weight", "eh_proj.weight"):
            return f"{mtp_pfx}.{suffix}"
        if suffix == "shared_head.norm.weight":
            return f"{mtp_pfx}.shared_head.norm.weight"
        # Echo aliases transformer Parameters to Layer 23, while Layer 24's
        # router correction bias is a distinct non-Parameter buffer.
        if suffix == "block_sparse_moe.e_score_correction_bias" and mtp_reuse_last_layer:
            return f"{mtp_pfx}.transformer_layer.mlp.gate.e_score_correction_bias"
        # Older exports nest the draft layer under `transformer_layer.`;
        # newer ones store it flat, exactly like a backbone layer.
        if suffix.startswith("transformer_layer."):
            suffix = suffix[len("transformer_layer."):]
        if mtp_reuse_last_layer:
            return None  # shared pointer, no weights to load
        return _map_layer_suffix(suffix, f"{mtp_pfx}.transformer_layer")

    # Backbone layer
    return _map_layer_suffix(suffix, layer_pfx)


def load_hf_safetensors_to_torch_model(
    model: Any,
    ckpt_dir: str,
    dtype: torch.dtype = torch.bfloat16,
    verbose: bool = False,
) -> dict:
    """Load HF safetensors shards into PyTorch model in-place.

    Weights are written into the model shard by shard (and converted straight to the
    target dtype) rather than accumulated in one fp32 dict: the full-model dict peaked
    around 40 GB of host RAM and got the process OOM-killed while training shared the box.
    """
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise ImportError("pip install safetensors") from exc

    weight_map = read_weight_map(ckpt_dir)

    cfg = model.config
    mtp_reuse = cfg.mtp_reuse_last_layer
    num_layers = cfg.num_hidden_layers

    # Build shard → keys map
    shard_to_keys: Dict[str, list] = {}
    for hk, shard in weight_map.items():
        shard_to_keys.setdefault(shard, []).append(hk)

    # Half-tensors waiting for their w1/w3 partner (may arrive in a later shard)
    expert_buf: Dict[str, Dict[str, torch.Tensor]] = {}
    shared_buf: Dict[str, Dict[str, torch.Tensor]] = {}

    model_sd = {k: v for k, v in model.state_dict().items()}
    model_keys = set(model_sd.keys())

    loaded_keys: set = set()
    updates: Dict[str, torch.Tensor] = {}
    stats = {"loaded": 0, "skipped": 0, "fused": 0, "shape_errors": [], "missing": []}
    shards = sorted(shard_to_keys.keys())
    logger.info("Loading %d shards …", len(shards))

    def flush():
        if not updates:
            return
        model.load_state_dict(updates, strict=False)
        loaded_keys.update(updates.keys())
        updates.clear()

    for i, shard_file in enumerate(shards):
        if (i + 1) % 20 == 0 or i == 0:
            logger.info("  shard %d/%d", i + 1, len(shards))
        shard_path = os.path.join(ckpt_dir, shard_file)

        with safe_open(shard_path, framework="pt") as sf:
            for hk in shard_to_keys[shard_file]:
                mapped = _map_hf_to_torch(hk, mtp_reuse, num_layers)

                if mapped is None:
                    stats["skipped"] += 1
                    continue

                arr = sf.get_tensor(hk)

                # ── Expert w1/w3 fusion ──────────────────────────────
                if mapped.startswith("__fuse_expert__"):
                    # e.g. __fuse_expert__model.layers.0__e0__w1w3__w1.weight
                    rest = mapped[len("__fuse_expert__"):]
                    layer_part, rest2 = rest.split("__e", 1)
                    eidx_part, sub_part = rest2.split("__w1w3__", 1)
                    buf_key = f"{layer_part}__e{eidx_part}"
                    wname = sub_part.split(".")[0]  # "w1" or "w3"
                    if buf_key not in expert_buf:
                        expert_buf[buf_key] = {}
                    expert_buf[buf_key][wname] = arr  # [moe_ffn, latent]

                    if "w1" in expert_buf[buf_key] and "w3" in expert_buf[buf_key]:
                        buf = expert_buf.pop(buf_key)
                        # w1: [moe_ffn, latent], w3: [moe_ffn, latent]
                        # up_gate_proj weight: [2*moe_ffn, latent] (concat w1, w3)
                        mkey = f"{layer_part}.mlp.experts.{eidx_part}.up_gate_proj.weight"
                        if mkey in model_keys:
                            updates[mkey] = torch.cat([buf["w1"], buf["w3"]], dim=0).to(
                                model_sd[mkey].dtype
                            )
                            stats["fused"] += 1
                    continue

                # ── Shared expert w1/w3 fusion ───────────────────────
                if mapped.startswith("__fuse_shared__"):
                    rest = mapped[len("__fuse_shared__"):]
                    layer_pfx, rest2 = rest.split("__w1w3__", 1)
                    wname = rest2.split(".")[0]
                    if layer_pfx not in shared_buf:
                        shared_buf[layer_pfx] = {}
                    shared_buf[layer_pfx][wname] = arr

                    if "w1" in shared_buf[layer_pfx] and "w3" in shared_buf[layer_pfx]:
                        buf = shared_buf.pop(layer_pfx)
                        mkey = f"{layer_pfx}.mlp.shared_experts.up_gate_proj.weight"
                        if mkey in model_keys:
                            updates[mkey] = torch.cat([buf["w1"], buf["w3"]], dim=0).to(
                                model_sd[mkey].dtype
                            )
                            stats["fused"] += 1
                    continue

                # ── Direct load ──────────────────────────────────────
                if mapped in model_keys:
                    expected_shape = list(model_sd[mapped].shape)
                    if list(arr.shape) != expected_shape:
                        stats["shape_errors"].append(
                            f"{hk} → {mapped}: expected {expected_shape}, got {list(arr.shape)}"
                        )
                        if verbose:
                            logger.warning("SHAPE MISMATCH: %s → %s: %s vs %s",
                                           hk, mapped, expected_shape, list(arr.shape))
                        continue
                    updates[mapped] = arr.to(model_sd[mapped].dtype)
                    stats["loaded"] += 1
                    if verbose:
                        logger.debug("LOAD %s → %s", hk, mapped)
                else:
                    stats["missing"].append(f"{hk} → {mapped}")
                    if verbose:
                        logger.warning("MISS %s → %s", hk, mapped)
        flush()

    flush()
    if expert_buf or shared_buf:
        logger.warning("Unpaired w1/w3 halves left over: %d expert, %d shared",
                       len(expert_buf), len(shared_buf))
    logger.info("Applied %d weight updates", len(loaded_keys))

    # Summary
    truly_missing = [k for k in model_keys
                     if k not in loaded_keys
                     # ECHO aliases the draft layer to Layer 23 — expected absent
                     and not (mtp_reuse and "transformer_layer" in k)
                     and "rotary_emb" not in k]        # buffers, not in checkpoint
    stats["missing_model_keys"] = truly_missing[:20]

    logger.info(
        "Done. loaded=%d fused=%d skipped=%d shape_errors=%d missing_model_keys=%d",
        stats["loaded"], stats["fused"], stats["skipped"],
        len(stats["shape_errors"]), len(truly_missing),
    )
    if stats["shape_errors"]:
        logger.warning("Shape errors (first 5): %s", stats["shape_errors"][:5])
    if truly_missing:
        logger.warning("Missing model keys (first 10): %s", truly_missing[:10])
    return stats


def build_and_load_torch(
    ckpt_dir: str,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    no_mtp: bool = False,
    preserve_mtp_sharing: bool = False,
):
    """Build + load in one call. Returns (model, config, stats)."""
    model, config = build_torch_model(
        ckpt_dir,
        device=device,
        dtype=dtype,
        no_mtp=no_mtp,
        preserve_mtp_sharing=preserve_mtp_sharing,
    )
    stats = load_hf_safetensors_to_torch_model(model, ckpt_dir, dtype=dtype)
    model.eval()
    return model, config, stats
