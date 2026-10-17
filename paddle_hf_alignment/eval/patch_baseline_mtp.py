#!/usr/bin/env python3
"""Load exported flat Layer 24 weights into the logical MTP decoder layer.

The historical filename is retained for compatibility. Both evaluated exports,
Echo and MoE-MTP baseline, materialize a complete independent Layer 24 using
flat keys such as ``model.layers.24.self_attn.*`` and
``model.layers.24.block_sparse_moe.*``. The training-time
``mtp_reuse_last_layer`` flag does not determine this exported layout.

The base loader handles MTP fusion modules but skips these flat transformer
keys. This module maps them through the established backbone mapping, retargets
the destination to ``model.mtp_heads.0.transformer_layer``, preserves the
existing w1/w3 fusion order, and also loads per-expert routed scaling.
"""
import json, os
import numpy as np
import torch
from modeling.torch_weight_loader import _map_hf_to_torch


def load_mtp_router_bias(model, ckpt_dir, num_layers=24, verbose=True):
    """Load Layer 24's draft-only router bias without mutating Backbone Layer 23."""
    from safetensors import safe_open

    key = f"model.layers.{num_layers}.block_sparse_moe.e_score_correction_bias"
    weight_map = json.load(
        open(os.path.join(ckpt_dir, "model.safetensors.index.json"))
    )["weight_map"]
    shard = weight_map.get(key)
    if shard is None:
        raise KeyError(f"Missing draft router bias: {key}")
    with safe_open(os.path.join(ckpt_dir, shard), framework="pt") as sf:
        bias = sf.get_tensor(key)

    mtp = model.model.mtp_heads[0]
    destination = mtp.transformer_layer.mlp.gate.e_score_correction_bias
    if tuple(destination.shape) != tuple(bias.shape):
        raise ValueError(
            f"Draft router bias shape mismatch: {tuple(destination.shape)} != {tuple(bias.shape)}"
        )
    destination.copy_(bias.to(device=destination.device, dtype=destination.dtype))
    if verbose:
        print(
            f"[mtp_router_bias] source=layer{num_layers} destination=draft-only "
            f"shape={tuple(destination.shape)} max_abs={destination.float().abs().max().item():.8f}"
        )
    return destination


def assert_alias_safe(model, updates, num_layers=24, verbose=True):
    """Refuse to write layer-N weights through an alias that also owns backbone layer N-1.

    With ``mtp_reuse_last_layer=True`` the HF port aliases the MTP layer's PARAMETERS to the
    last backbone layer (``_alias_module_parameters``), i.e. they are the same tensor objects.
    ``load_state_dict`` therefore writes ``model.layers.{N-1}.*`` as a side effect of writing
    ``model.mtp_heads.0.transformer_layer.*``.

    On an Echo checkpoint this is harmless: training aliased the same parameters, so the
    exported layer 24 body is bit-identical to layer 23 (verified: routed_scaling_factor_param,
    block_sparse_moe.gate.weight, self_attn.*, layernorms, experts, shared_experts all
    ``max|d| = 0``).  On any checkpoint where they differ it would silently corrupt the
    backbone.  Buffers (``e_score_correction_bias``) are NOT aliased and are always safe.

    Returns the number of aliased keys checked.  Raises ValueError on the first real conflict
    unless TARGET_ALLOW_ALIAS_WRITE=1 (then it warns and lets the write happen).
    """
    # remove_duplicate=False is REQUIRED: named_parameters() de-duplicates shared tensors by
    # default, so an aliased MTP parameter simply would not appear and the check would silently
    # pass on exactly the models it is meant to protect.
    params = dict(model.named_parameters(remove_duplicate=False))
    prefix = "model.mtp_heads.0.transformer_layer."
    backbone = f"model.layers.{num_layers - 1}."
    checked, conflicts = 0, []
    for key, new_val in updates.items():
        if not key.startswith(prefix):
            continue
        src = params.get(key)
        twin = params.get(backbone + key[len(prefix):])
        if src is None or twin is None or src is not twin:
            continue                      # not aliased -> writing it is private to the MTP layer
        checked += 1
        cur = src.detach().to("cpu", torch.float32)
        new = new_val.detach().to("cpu", torch.float32)
        if cur.shape != new.shape or not torch.equal(cur, new):
            d = float((cur - new).abs().max()) if cur.shape == new.shape else float("nan")
            conflicts.append((key, d))
    if conflicts and os.environ.get("TARGET_ALLOW_ALIAS_WRITE") != "1":
        head = ", ".join(f"{k} (max|d|={d:.6g})" for k, d in conflicts[:3])
        raise ValueError(
            f"patch_baseline_mtp would overwrite backbone layer {num_layers - 1} through the MTP "
            f"alias: {len(conflicts)}/{checked} aliased tensors differ from the checkpoint's "
            f"layer-{num_layers} copy. First: {head}. This checkpoint's MTP body is NOT a copy of "
            f"backbone-last, so it needs an un-aliased MTP layer (mtp_reuse_last_layer=False). "
            f"Set TARGET_ALLOW_ALIAS_WRITE=1 only if clobbering layer "
            f"{num_layers - 1} is intended."
        )
    if conflicts:
        print(f"[patch_baseline_mtp] WARNING: writing {len(conflicts)} aliased tensors that "
              f"differ -> backbone layer {num_layers - 1} will change (override in effect)")
    elif verbose:
        print(f"[patch_baseline_mtp] alias check: {checked} aliased tensors identical to the "
              f"layer-{num_layers} export (safe to write)")
    return checked


def patch_baseline_mtp_weights(model, ckpt_dir, num_layers=24, verbose=True):
    """Load flat layer-24 transformer weights into the model's MTP layer.

    No-op when the model was built with independent MTP layers
    (``mtp_reuse_last_layer=False``): ``torch_weight_loader`` then already loads every
    depth's flat keys into ``model.mtp_heads.{k}.transformer_layer``, including depths
    1..D-1 which this function never covered. Kept for models still built with the
    aliased Echo layout.
    """
    if not getattr(model.config, "mtp_reuse_last_layer", False):
        if verbose:
            print("[patch_baseline_mtp] skipped: MTP layers already loaded per depth "
                  "by torch_weight_loader (mtp_reuse_last_layer=False)")
        return 0

    from safetensors import safe_open
    weight_map = json.load(open(os.path.join(ckpt_dir, "model.safetensors.index.json")))["weight_map"]
    model_sd = model.state_dict()
    model_keys = set(model_sd.keys())
    SRC = f"model.layers.{num_layers}."          # e.g. model.layers.24.
    STAND = "model.layers.23"                     # pretend backbone layer 23
    DST = "model.mtp_heads.0.transformer_layer"

    # collect target flat keys (attn / moe / layernorms only)
    def is_sublayer(suffix):
        return (suffix.startswith("self_attn.") or suffix.startswith("block_sparse_moe.")
                or suffix in ("input_layernorm.weight", "post_attention_layernorm.weight"))

    shard_to_keys = {}
    for hk, shard in weight_map.items():
        if hk.startswith(SRC) and is_sublayer(hk[len(SRC):]):
            shard_to_keys.setdefault(shard, []).append(hk)

    updates, expert_buf, shared_buf = {}, {}, {}
    n_loaded = n_fused = n_miss = 0
    for shard, hks in shard_to_keys.items():
        with safe_open(os.path.join(ckpt_dir, shard), framework="pt") as sf:
            for hk in hks:
                suffix = hk[len(SRC):]
                mapped = _map_hf_to_torch(f"{STAND}.{suffix}", False, num_layers)
                if mapped is None:
                    continue
                mapped = mapped.replace(STAND, DST)   # retarget prefix
                arr = sf.get_tensor(hk).to(torch.float32).numpy()

                if mapped.startswith("__fuse_expert__"):
                    rest = mapped[len("__fuse_expert__"):]
                    layer_part, rest2 = rest.split("__e", 1)
                    eidx_part, sub_part = rest2.split("__w1w3__", 1)
                    buf_key = f"{layer_part}__e{eidx_part}"
                    wname = sub_part.split(".")[0]
                    expert_buf.setdefault(buf_key, {})[wname] = arr
                    if "w1" in expert_buf[buf_key] and "w3" in expert_buf[buf_key]:
                        buf = expert_buf.pop(buf_key)
                        fused = np.concatenate([buf["w1"], buf["w3"]], axis=0)
                        mkey = f"{layer_part}.mlp.experts.{eidx_part}.up_gate_proj.weight"
                        if mkey in model_keys:
                            updates[mkey] = torch.from_numpy(fused).to(model_sd[mkey].dtype); n_fused += 1
                    continue

                if mapped.startswith("__fuse_shared__"):
                    rest = mapped[len("__fuse_shared__"):]
                    layer_pfx, rest2 = rest.split("__w1w3__", 1)
                    wname = rest2.split(".")[0]
                    shared_buf.setdefault(layer_pfx, {})[wname] = arr
                    if "w1" in shared_buf[layer_pfx] and "w3" in shared_buf[layer_pfx]:
                        buf = shared_buf.pop(layer_pfx)
                        fused = np.concatenate([buf["w1"], buf["w3"]], axis=0)
                        mkey = f"{layer_pfx}.mlp.shared_experts.up_gate_proj.weight"
                        if mkey in model_keys:
                            updates[mkey] = torch.from_numpy(fused).to(model_sd[mkey].dtype); n_fused += 1
                    continue

                if mapped in model_keys:
                    if list(arr.shape) == list(model_sd[mapped].shape):
                        updates[mapped] = torch.from_numpy(arr).to(model_sd[mapped].dtype); n_loaded += 1
                    else:
                        n_miss += 1
                else:
                    n_miss += 1

    scaling_key = (
        f"model.layers.{num_layers}.transformer_layer."
        "mlp.gate.routed_scaling_factor_param"
    )
    scaling_shard = weight_map.get(scaling_key)
    if scaling_shard is not None:
        with safe_open(os.path.join(ckpt_dir, scaling_shard), framework="pt") as sf:
            scaling = sf.get_tensor(scaling_key)
        model_key = f"{DST}.mlp.gate.routed_scaling_factor_param"
        if model_key in model_keys and tuple(model_sd[model_key].shape) == tuple(scaling.shape):
            updates[model_key] = scaling.to(dtype=model_sd[model_key].dtype)
            n_loaded += 1
        else:
            n_miss += 1

    assert_alias_safe(model, updates, num_layers, verbose)
    model.load_state_dict(updates, strict=False)
    if verbose:
        print(f"[patch_baseline_mtp] loaded={n_loaded} fused={n_fused} miss={n_miss} "
              f"(updates={len(updates)})")
    return len(updates)