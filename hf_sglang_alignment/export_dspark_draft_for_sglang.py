"""Export a trained the target DSpark draft checkpoint into an sglang-servable draft.

Source: SpecForge `TargetDSparkDraftModel` (ECHO/run4n_dspark/epoch_*_step_*),
a 5-layer the target MLA + latent-MoE backbone plus the DSpark heads:

    layers.{0..4}.*                 the target decoder layer (same layout as the
                                    target's MTP block, so the same transforms
                                    as export_eagle3_draft_for_sglang_v2.py)
    fc.weight [H, K*H]              DFlash target-context projector (K = number
                                    of captured target layers, here 5)
    hidden_norm.weight, norm.weight
    markov_head.markov_w{1,2}       vanilla Markov head, rank 256
    confidence_head.proj.{weight,bias}

Unlike the EAGLE-3 export there is no prefix remap and no `fc` transpose:
sglang's DFlashDraftModel keeps `layers.*` / `fc` / `hidden_norm` / `norm` at the
top level and applies `fc` as a Linear, not as a matmul against a raw tensor.

Transforms:
  1. experts        _grouped_w13[e] -> experts.e.{gate,up}_proj (row split at I)
                    _grouped_w2[e]  -> experts.e.down_proj (no transpose)
  2. shared expert  up_gate_proj -> shared_experts.{gate,up}_proj (row split)
  3. renames        self_attn.gate_proj -> self_attn.attn_out_gate
                    mlp.gate.routed_scaling_factor_param -> mlp.routed_scaling
  4. rope regroup   q_b_proj per-head rope rows and kv_a_proj_with_mqa k_pe rows
                    permuted [0,2,..,1,3,..] (the target rope_interleave=False)
  5. dtypes         e_score_correction_bias -> fp32 (bf16 was measured to send
                    59% of tokens to the wrong experts), routed_scaling bf16

Usage:
  python3 export_dspark_draft_for_sglang.py \
      --ckpt run4n_dspark/epoch_1_step_44392 \
      --out  serve_dspark_s44392
"""

import argparse
import json
import os
import re

import torch
from safetensors import safe_open
from safetensors.torch import save_file

SUBSTR_RENAME = {
    "self_attn.gate_proj": "self_attn.attn_out_gate",
    "mlp.gate.routed_scaling_factor_param": "mlp.routed_scaling",
}
PASSTHROUGH = (
    "fc.weight",
    "hidden_norm.weight",
    "norm.weight",
)
# DSpark-only heads. A DFlash checkpoint (model_type target_dflash_draft) is exactly
# this architecture MINUS these four tensors -- same 5-layer body, same fc/norms --
# which is what makes DFlash vs DSpark the clean ablation of the first-order Markov
# chain over draft positions. So they are optional, not required.
PASSTHROUGH_DSPARK = (
    "markov_head.markov_w1.weight",
    "markov_head.markov_w2.weight",
    "confidence_head.proj.weight",
    "confidence_head.proj.bias",
)
LAYER_RE = re.compile(r"^layers\.(\d+)\.(.*)$")


def rope_regroup(w, start, end, step):
    """Permute rows [start:end) of each `step`-sized block to even-then-odd order.

    the target's `_apply_rotary(interleave=False)` does cat([x[0::2], x[1::2]]) before
    the split-half rotation; sglang's neox rotary does not, so the permutation is
    folded into the weights once at export.
    """
    out = w.clone()
    n = end - start
    perm = torch.cat([torch.arange(0, n, 2), torch.arange(1, n, 2)])
    for base in range(0, w.shape[0], step):
        s, e = base + start, base + end
        if e > w.shape[0]:
            break
        out[s:e] = w[s:e][perm]
    return out


def scan_finite(tensors, what):
    bad = [
        k
        for k, v in tensors.items()
        if v.is_floating_point() and not torch.isfinite(v).all()
    ]
    if bad:
        raise SystemExit(f"{what}: {len(bad)} non-finite tensors, e.g. {bad[:5]}")
    print(f"  isfinite {what}: {len(tensors)} tensors OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    src = {}
    with safe_open(os.path.join(a.ckpt, "model.safetensors"), framework="pt") as sf:
        for k in sf.keys():
            src[k] = sf.get_tensor(k)
    print(f"source: {len(src)} tensors from {a.ckpt}")
    scan_finite(src, "source")

    dcfg = json.load(open(os.path.join(a.ckpt, "config.json")))
    n_layers = int(dcfg["num_hidden_layers"])
    n_exp = int(dcfg["n_routed_experts"])
    qk_nope = int(dcfg["qk_nope_head_dim"])
    qk_rope = int(dcfg["qk_rope_head_dim"])
    kv_lora = int(dcfg["kv_lora_rank"])

    out = {}
    for k in PASSTHROUGH:
        if k not in src:
            raise SystemExit(f"missing source key {k!r}")
        out[k] = src[k].contiguous()

    have_markov = all(k in src for k in PASSTHROUGH_DSPARK)
    if have_markov:
        for k in PASSTHROUGH_DSPARK:
            out[k] = src[k].contiguous()
    else:
        missing = [k for k in PASSTHROUGH_DSPARK if k not in src]
        if len(missing) != len(PASSTHROUGH_DSPARK):
            raise SystemExit(f"partial DSpark heads, refusing to guess: missing {missing}")
        print("  no markov/confidence heads -> exporting as DFlash (no draft-position chain)")

    consumed = set(PASSTHROUGH) | set(PASSTHROUGH_DSPARK)
    for layer in range(n_layers):
        p = f"layers.{layer}."
        # 1. routed experts: stacked -> per-expert
        w13, w2 = src[p + "mlp._grouped_w13"], src[p + "mlp._grouped_w2"]
        inter = w13.shape[1] // 2
        for e in range(n_exp):
            out[f"{p}mlp.experts.{e}.gate_proj.weight"] = w13[e][:inter].contiguous()
            out[f"{p}mlp.experts.{e}.up_proj.weight"] = w13[e][inter:].contiguous()
            out[f"{p}mlp.experts.{e}.down_proj.weight"] = w2[e].contiguous()
        # 2. shared expert: fused up_gate -> gate + up
        ug = src[p + "mlp.shared_experts.up_gate_proj.weight"]
        half = ug.shape[0] // 2
        out[f"{p}mlp.shared_experts.gate_proj.weight"] = ug[:half].contiguous()
        out[f"{p}mlp.shared_experts.up_proj.weight"] = ug[half:].contiguous()
        consumed |= {
            p + "mlp._grouped_w13",
            p + "mlp._grouped_w2",
            p + "mlp.shared_experts.up_gate_proj.weight",
        }

    # 3./4./5. everything else: renames, rope regroup, dtype rules
    for k, v in src.items():
        if k in consumed:
            continue
        m = LAYER_RE.match(k)
        if m is None:
            raise SystemExit(f"unmapped source key {k!r} -- update the weight map")
        tail = m.group(2)
        for old, new in SUBSTR_RENAME.items():
            if tail.startswith(old):
                tail = new + tail[len(old) :]
                break
        nk = f"layers.{m.group(1)}.{tail}"
        if nk.endswith("e_score_correction_bias"):
            v = v.float()
        elif nk.endswith("routed_scaling"):
            v = v.to(torch.bfloat16)
        if nk.endswith("self_attn.q_b_proj.weight"):
            v = rope_regroup(v, qk_nope, qk_nope + qk_rope, qk_nope + qk_rope)
        elif nk.endswith("self_attn.kv_a_proj_with_mqa.weight"):
            v = rope_regroup(v, kv_lora, kv_lora + qk_rope, v.shape[0])
        out[nk] = v.contiguous()

    # ---- guards -----------------------------------------------------------
    n_expert_keys = sum(1 for k in out if ".experts." in k)
    if n_expert_keys != 3 * n_exp * n_layers:
        raise SystemExit(f"expert keys {n_expert_keys} != 3*{n_exp}*{n_layers}")
    for layer in range(n_layers):
        bias = out[f"layers.{layer}.mlp.gate.e_score_correction_bias"]
        if bias.dtype != torch.float32:
            raise SystemExit(f"layer {layer} correction bias must be fp32")
    if int(out["fc.weight"].shape[1]) % int(dcfg["hidden_size"]) != 0:
        raise SystemExit("fc.weight in_features is not a multiple of hidden_size")
    scan_finite(out, "product")

    # ---- config -----------------------------------------------------------
    cfg = dict(dcfg)
    # A markov-less (DFlash) export keeps architectures=TargetDSparkModel /
    # model_type=target_dspark_draft on purpose and is served through the DSPARK
    # algorithm as a documented degraded mode (`model_runner.py:553`: markov_rank=0 ->
    # "serving as DFlash (no markov correction, no confidence head)"). Writing the
    # honest-looking `target_dflash_draft` instead breaks loading, because sglang
    # resolves the draft config through AutoConfig and only `target_dspark_draft` is
    # known there: KeyError: 'target_dflash_draft' (hit 2026-09-23 02:45).
    cfg.update(
        {
            "architectures": ["TargetDSparkModel"],
            "model_type": "target_dspark_draft",
            "torch_dtype": "bfloat16",
            "_exported_from": a.ckpt,
            "_markov_heads": bool(have_markov),
        }
    )

    os.makedirs(a.out, exist_ok=True)
    save_file(out, os.path.join(a.out, "model.safetensors"), metadata={"format": "pt"})
    json.dump(cfg, open(os.path.join(a.out, "config.json"), "w"), indent=2)
    n_ctx = int(out["fc.weight"].shape[1]) // int(dcfg["hidden_size"])
    print(f"exported {len(out)} tensors -> {a.out}")
    print(
        f"  layers={n_layers} experts={n_exp} context_features={n_ctx} "
        f"block_size={dcfg.get('block_size')} "
        f"target_layer_ids={dcfg.get('dspark_config', {}).get('target_layer_ids')}"
    )


if __name__ == "__main__":
    main()

