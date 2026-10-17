"""Warm-start the target EAGLE-3 draft from Echo's own pretrained MTP block.

Checkpoint layout (verified against echo1.5_sft_step5000, 2026-09-18):

* 27 stored layers: 0-23 backbone, 24/25/26 = the three MTP depths.
* ``mtp_reuse_last_layer=True``, and the checkpoint honours it literally: every
  BODY tensor of layer 24 is **bit-identical** to layer 23 (attention, MoE
  experts, ``gate.weight``, layernorms, ``routed_scaling_factor_param`` all show
  ``max|d| == 0``).
* The one exception is ``block_sparse_moe.e_score_correction_bias``:
  layer 24 differs from layer 23 by ``max|d| = 0.108``. It is the only
  depth-specific tensor, which is why ``TargetMTPHead`` keeps a distinct
  module for non-Parameter buffers while aliasing the Parameters. Given that a
  mere 0.002 bf16 perturbation of this bias flipped the top-8 choice for 59% of
  layer-0 tokens, taking layer 23's bias here would silently change the router.
  **The bias must come from the MTP layer, not from the body layer.**
* ``gate.weight``, ``e_score_correction_bias`` and ``routed_scaling_factor_param``
  are fp32 in the checkpoint and must stay fp32 (same trap as
  ``torch_weight_loader``'s ``.to(dtype)`` casting float buffers).

What is NOT in the checkpoint and therefore initialized here:

* ``fc`` (3*1024 -> 1024), the EAGLE-3 three-route fusion: set to ``[0 | 0 | I]``
  so that at step 0 ``fc([h_low; h_mid; h_high]) == h_high``, i.e. the
  warm-started MTP block sees exactly its pretraining input. This is only true
  if the third capture route IS the final backbone hidden, so the warm-start run
  must capture ``[..., num_hidden_layers - 1]`` (layer 23), not the cold-start
  default ``[1, 11, 20]``.
* ``lm_head`` (32000 x 1024): taken from the target's ``lm_head`` rows selected
  by the draft vocabulary mapping (``d2t``), not random.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List

import torch
from safetensors import safe_open

_EXPERT_W13 = ("w1", "w3")


def _index(model_path: str) -> Dict[str, str]:
    with open(os.path.join(model_path, "model.safetensors.index.json")) as f:
        return json.load(f)["weight_map"]


class _Reader:
    """Cache one open handle per shard; the MTP layer spans several shards."""

    def __init__(self, model_path: str, weight_map: Dict[str, str]):
        self._path = model_path
        self._map = weight_map
        self._open: Dict[str, object] = {}

    def get(self, key: str) -> torch.Tensor:
        shard = self._map[key]
        fh = self._open.get(shard)
        if fh is None:
            fh = safe_open(os.path.join(self._path, shard), framework="pt")
            self._open[shard] = fh
        return fh.get_tensor(key)

    def has(self, key: str) -> bool:
        return key in self._map


# draft parameter name (under mtp_head.) -> suffix under model.layers.<BODY>.
_BODY_MAP = {
    "transformer_layer.input_layernorm.weight": "input_layernorm.weight",
    "transformer_layer.post_attention_layernorm.weight": "post_attention_layernorm.weight",
    "transformer_layer.self_attn.q_a_proj.weight": "self_attn.q_a_proj.weight",
    "transformer_layer.self_attn.q_a_layernorm.weight": "self_attn.q_a_layernorm.weight",
    "transformer_layer.self_attn.q_b_proj.weight": "self_attn.q_b_proj.weight",
    "transformer_layer.self_attn.kv_a_proj_with_mqa.weight": "self_attn.kv_a_proj_with_mqa.weight",
    "transformer_layer.self_attn.kv_a_layernorm.weight": "self_attn.kv_a_layernorm.weight",
    "transformer_layer.self_attn.kv_b_proj.weight": "self_attn.kv_b_proj.weight",
    "transformer_layer.self_attn.o_proj.weight": "self_attn.o_proj.weight",
    "transformer_layer.self_attn.gate_proj.weight": "self_attn.gate_proj.weight",
    "transformer_layer.mlp.gate.weight": "block_sparse_moe.gate.weight",
    "transformer_layer.mlp.fc1_latent_proj.weight": "block_sparse_moe.fc1_latent_proj.weight",
    "transformer_layer.mlp.fc2_latent_proj.weight": "block_sparse_moe.fc2_latent_proj.weight",
    "transformer_layer.mlp.shared_experts.down_proj.weight": "block_sparse_moe.shared_experts.w2.weight",
}
# these live only on the MTP layer
_MTP_ONLY_MAP = {
    "enorm.weight": "enorm.weight",
    "hnorm.weight": "hnorm.weight",
    "eh_proj.weight": "eh_proj.weight",
    "shared_head.norm.weight": "shared_head.norm.weight",
}
# depth-specific and fp32: must come from the MTP layer, never the body layer
_MTP_FP32_MAP = {
    "transformer_layer.mlp.gate.e_score_correction_bias": "block_sparse_moe.e_score_correction_bias",
}
_BODY_FP32_MAP = {
    "transformer_layer.mlp.gate.routed_scaling_factor_param": "block_sparse_moe.gate.routed_scaling_factor_param",
}


def _assign(dst: torch.Tensor, src: torch.Tensor, name: str, keep_fp32: bool) -> None:
    if tuple(dst.shape) != tuple(src.shape):
        raise ValueError(f"{name}: draft {tuple(dst.shape)} vs ckpt {tuple(src.shape)}")
    with torch.no_grad():
        if keep_fp32:
            dst.copy_(src.float())
        else:
            dst.copy_(src.to(dst.dtype))


def _coerce_router_fp32(draft) -> List[str]:
    """Force the router CORRECTION BIAS back to fp32, and only that one.

    The trainer builds the draft with ``torch_dtype=bfloat16``, and ``.to(dtype)``
    casts float buffers as well. bf16 resolution near 0.6 is ~0.004 while the
    sigmoid scores the bias is *added* to are O(1e-4..1e-1): a 0.002 perturbation
    was measured to flip the top-8 expert set for 59% of layer-0 tokens, so this
    tensor must stay fp32.

    ``routed_scaling_factor_param`` is deliberately left in bf16: it is a
    trainable ``nn.Parameter``, and FSDP flattens parameters into one buffer
    ("Must flatten tensors with uniform dtype but got torch.bfloat16 and
    torch.float32"). It is also a *multiplicative* factor applied after top-k
    normalisation, so bf16's ~0.4% relative error is proportional noise and
    cannot reorder the expert selection. ``e_score_correction_bias`` is a buffer,
    which FSDP does not flatten, so keeping it fp32 is free.
    """
    gate = draft.mtp_head.transformer_layer.mlp.gate
    touched = []
    name = "e_score_correction_bias"
    t = getattr(gate, name, None)
    if isinstance(t, torch.Tensor) and not isinstance(t, torch.nn.Parameter):
        if t.dtype != torch.float32:
            setattr(gate, name, t.float())
            touched.append(name)
    elif isinstance(t, torch.nn.Parameter):
        raise TypeError(
            f"{name} is a Parameter in this draft; keeping it fp32 would break FSDP "
            "flattening. Revisit: it must be fp32 for correct routing."
        )
    return touched


def warmstart_target_mtp_eagle(
    draft,
    target_model_path: str,
    *,
    mtp_depth: int = 0,
    init_fc_identity: bool = True,
    init_lm_head_from_target: bool = True,
) -> Dict[str, object]:
    """Copy Echo's pretrained MTP block into the EAGLE-3 draft, in place.

    Returns a report dict; every mapped tensor is verified, and the two fp32
    router tensors are re-read and compared bitwise afterwards.
    """
    weight_map = _index(target_model_path)
    reader = _Reader(target_model_path, weight_map)
    # Read the depth from the TARGET config: the draft config's num_hidden_layers
    # is the draft's own depth (1), not the target's.
    with open(os.path.join(target_model_path, "config.json")) as f:
        tgt_cfg = json.load(f)
    n_layers = int(tgt_cfg["num_hidden_layers"])
    n_mtp = int(tgt_cfg.get("num_nextn_predict_layers", 1))
    if not 0 <= int(mtp_depth) < n_mtp:
        raise ValueError(f"mtp_depth {mtp_depth} outside the checkpoint's {n_mtp} MTP layers")
    mtp = n_layers + int(mtp_depth)     # 24: depth-0 MTP layer
    # Where the transformer BODY (attention, router, shared + routed experts) lives
    # depends on mtp_reuse_last_layer:
    #   true  (echo1.5_sft): the MTP block aliases decoder layer n_layers-1, and layer
    #         24 carries only enorm/hnorm/eh_proj/shared_head -- so the body must come
    #         from 23.
    #   false (ds_mtp1_sft): layer 24 is a STANDALONE MTP block that owns its own body
    #         (verified: layer 24 has 790 tensors incl. self_attn.o_proj,
    #         block_sparse_moe.{gate,experts.*,shared_experts.*}, while layer 23 has 786
    #         and no enorm/hnorm/eh_proj). Taking the body from 23 here would warm-start
    #         the draft from an ORDINARY decoder layer and silently lose the trained MTP
    #         block -- a warm start in name only.
    reuse_last = bool(tgt_cfg.get("mtp_reuse_last_layer", False))
    body = (n_layers - 1) if reuse_last else mtp

    coerced = _coerce_router_fp32(draft)

    params = dict(draft.named_parameters())
    buffers = dict(draft.named_buffers())
    slots = {**params, **buffers}
    report: Dict[str, object] = {
        "body_layer": body, "mtp_layer": mtp, "mtp_reuse_last_layer": reuse_last,
        "loaded": 0, "experts": 0,
        "router_fp32_coerced": coerced, "checks": []
    }

    def slot(name: str) -> torch.Tensor:
        full = f"mtp_head.{name}"
        if full not in slots:
            raise KeyError(f"draft has no {full}")
        return slots[full]

    for dst_name, suffix in _MTP_ONLY_MAP.items():
        _assign(slot(dst_name), reader.get(f"model.layers.{mtp}.{suffix}"), dst_name, False)
        report["loaded"] += 1
    for dst_name, suffix in _BODY_MAP.items():
        _assign(slot(dst_name), reader.get(f"model.layers.{body}.{suffix}"), dst_name, False)
        report["loaded"] += 1
    for dst_name, suffix in _BODY_FP32_MAP.items():
        _assign(slot(dst_name), reader.get(f"model.layers.{body}.{suffix}"), dst_name, True)
        report["loaded"] += 1
    for dst_name, suffix in _MTP_FP32_MAP.items():
        _assign(slot(dst_name), reader.get(f"model.layers.{mtp}.{suffix}"), dst_name, True)
        report["loaded"] += 1

    # shared expert w1/w3 -> fused up_gate_proj [2*I, H]
    w1 = reader.get(f"model.layers.{body}.block_sparse_moe.shared_experts.w1.weight")
    w3 = reader.get(f"model.layers.{body}.block_sparse_moe.shared_experts.w3.weight")
    _assign(
        slot("transformer_layer.mlp.shared_experts.up_gate_proj.weight"),
        torch.cat([w1, w3], dim=0),
        "shared_experts.up_gate_proj.weight",
        False,
    )
    report["loaded"] += 1

    # 256 routed experts -> the stacked trainable Parameters
    g13 = slot("transformer_layer.mlp._grouped_w13")
    g2 = slot("transformer_layer.mlp._grouped_w2")
    n_experts = g13.shape[0]
    with torch.no_grad():
        for e in range(n_experts):
            base = f"model.layers.{body}.block_sparse_moe.experts.{e}"
            ew1 = reader.get(f"{base}.w1.weight")
            ew3 = reader.get(f"{base}.w3.weight")
            ew2 = reader.get(f"{base}.w2.weight")
            g13[e].copy_(torch.cat([ew1, ew3], dim=0).to(g13.dtype))
            g2[e].copy_(ew2.to(g2.dtype))
            report["experts"] += 1

    if init_fc_identity:
        fc = draft.fc.weight  # [hidden, 3*hidden]
        h = fc.shape[0]
        n_routes = fc.shape[1] // h
        with torch.no_grad():
            fc.zero_()
            fc[:, (n_routes - 1) * h:].copy_(torch.eye(h, dtype=fc.dtype, device=fc.device))
        report["fc_init"] = f"[0]*{n_routes - 1} | I_{h}"

    if init_lm_head_from_target:
        d2t = buffers.get("d2t")
        if d2t is None:
            raise KeyError("draft has no d2t buffer")
        draft_ids = torch.arange(d2t.numel(), device=d2t.device) + d2t
        # Observability only, deliberately non-blocking. train_eagle3.py calls
        # load_vocab_mapping() AFTER this function, so whether d2t already holds the
        # dataset-derived mapping here depends on the run (fresh vs resumed). If it is
        # still all zeros, the rows taken below are target ids 0..31999 instead of the
        # hot vocab and nothing downstream complains -- so record the evidence rather
        # than reasoning about it after the fact.
        report["d2t_abs_sum_at_lm_head_init"] = int(d2t.abs().sum().item())
        report["hot_ids_head"] = draft_ids[:8].tolist()
        tgt_head = reader.get("lm_head.weight")
        with torch.no_grad():
            draft.lm_head.weight.copy_(
                tgt_head[draft_ids.cpu()].to(draft.lm_head.weight.dtype)
            )
        report["lm_head_init"] = f"target rows via d2t ({d2t.numel()} of {tgt_head.shape[0]})"

    # post-conditions: the two fp32 router tensors must match the checkpoint bitwise
    for dst_name, suffix, layer in (
        (list(_MTP_FP32_MAP)[0], list(_MTP_FP32_MAP.values())[0], mtp),
        (list(_BODY_FP32_MAP)[0], list(_BODY_FP32_MAP.values())[0], body),
    ):
        dst = slot(dst_name)
        got = dst.float().cpu()
        want = reader.get(f"model.layers.{layer}.{suffix}").float()
        diff = (got - want).abs().max().item()
        # fp32 buffers must match bitwise; the bf16 Parameter only to bf16 resolution
        # (it is cast on purpose -- see _coerce_router_fp32).
        tol = 0.0 if dst.dtype == torch.float32 else (want.abs().max().item() * 2 ** -8)
        report["checks"].append(
            f"{dst_name.split('.')[-1]} from layer{layer}: max|d|={diff:g} "
            f"dtype={dst.dtype} tol={tol:g}"
        )
        if diff > tol:
            raise ValueError(f"{dst_name} did not load (max|d|={diff} > tol={tol})")
    return report
