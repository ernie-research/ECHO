"""Export a trained the target EAGLE-3 draft checkpoint into an sglang-servable draft.

Rebuild of the original export script, which was truncated by a disk ENOSPC on
2026-09-19 00:52 (181 lines -> 46, export logic gone, no copy, ECHO is not a git
repo). Every rule below was re-derived by diffing the surviving artifact
`serve_eagle3_cold/` (794 tensors) against its source checkpoint, and is written
down in ECHO/EAGLE3_EXPORT_WEIGHT_MAP_20260919.md so a second loss is a lookup,
not an archaeology project.

  python3 export_eagle3_draft_for_sglang_v2.py \
      --ckpt run4n_eagle3_cold/epoch_0_step_11838 \
      --out  serve_eagle3_cold_s11838

Transforms (see the md for the measurements that pin each one):
  1. prefix remap  mtp_head.transformer_layer.* -> model.layers.0.*
                   mtp_head.{enorm,hnorm,eh_proj,shared_head.norm} -> model.layers.0.*
                   (the nextn loader rewrites model.layers.0 -> model / model.decoder)
                   lm_head.weight -> model.shared_head.head.weight
  2. renames       self_attn.gate_proj -> self_attn.attn_out_gate
                   mlp.gate.routed_scaling_factor_param -> mlp.routed_scaling
  3. transpose     fc.weight [H, 3H] -> model.rot_weight [3H, H]
                   (DeepseekModelNextN.forward does matmul(hidden, rot_weight))
  4. experts       _grouped_w13[e] -> experts.e.{gate,up}_proj (row split at I)
                   _grouped_w2[e]  -> experts.e.down_proj (no transpose)
  5. shared expert up_gate_proj -> shared_experts.{gate,up}_proj (row split at I)
  6. rope regroup  q_b_proj per-head rope rows and kv_a_proj_with_mqa k_pe rows
                   permuted [0,2,..,1,3,..] (the target interleave=False layout)
  7. embed_tokens  not exported; serving loads it from the target

Guards this version adds (the original had none of them; each failure mode below
is silent -- the draft serves with wrong or uninitialised weights):
  * isfinite scan of source AND product (a NaN checkpoint loads fine and serves
    garbage; BF16Optimizer used to have no non-finite guard)
  * required serving keys present
  * expert count == 3 * n_routed_experts
  * dtype assertions: e_score_correction_bias fp32, routed_scaling bf16
  * no leftover trainer prefixes
"""

import argparse
import json
import os

import torch
from safetensors import safe_open
from safetensors.torch import save_file

TL = "mtp_head.transformer_layer."
# The DeepSeek nextn loader keeps ONLY keys under
#   nextn_layer_prefix = f"model.layers.{0 if num_hidden_layers == 1 else N}"
# and silently `continue`s on everything else (deepseek_weight_loader.py:395-402,
# :183-184). It then rewrites that prefix to "model" for the four spec names and
# to "model.decoder" for the rest. So the export has to speak that naming, or the
# whole draft body loads as random init with no error at all -- which is exactly
# what happened before 2026-09-19 (accept_len stuck at 1.0000).
NEXTN = "model.layers.0."
HEAD_RENAME = {
    "mtp_head.enorm.weight": NEXTN + "enorm.weight",
    "mtp_head.hnorm.weight": NEXTN + "hnorm.weight",
    "mtp_head.eh_proj.weight": NEXTN + "eh_proj.weight",
    "mtp_head.shared_head.norm.weight": NEXTN + "shared_head.norm.weight",
    # The loader skips `shared_head.head` (native MTP borrows the target's head);
    # EAGLE-3 has its own truncated head, so TargetMTPEagle3ForCausalLM.load_weights
    # intercepts this key and loads it directly.
    "lm_head.weight": "model.shared_head.head.weight",
}

SUBSTR_RENAME = {
    "self_attn.gate_proj": "self_attn.attn_out_gate",
    "mlp.gate.routed_scaling_factor_param": "mlp.routed_scaling",
}
REQUIRED = (
    "model.rot_weight",
    "model.shared_head.head.weight",
    NEXTN + "shared_head.norm.weight",
    NEXTN + "enorm.weight",
    NEXTN + "hnorm.weight",
    NEXTN + "eh_proj.weight",
    NEXTN + "mlp.gate.e_score_correction_bias",
    NEXTN + "mlp.routed_scaling",
    "d2t",
    "t2d",
)



def rope_regroup(w, start, end, step):
    """Permute rows [start:end) of each `step`-sized block to even-then-odd order.

    the target's `_apply_rotary(interleave=False)` does cat([x[0::2], x[1::2]]) before
    the split-half rotation; sglang's neox rotary does not, so the permutation is
    folded into the weights once at export (PORT doc 5.2).
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
    bad = [k for k, v in tensors.items()
           if v.is_floating_point() and not torch.isfinite(v).all()]
    if bad:
        raise SystemExit(f"{what}: {len(bad)} non-finite tensors, e.g. {bad[:5]}")
    print(f"  isfinite {what}: {len(tensors)} tensors OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--target-config",
                    default="<WORKSPACE>/echo1.5_sft_step5000/config.json")
    ap.add_argument("--aux-layer-ids", default="",
                    help="comma list; default = keep the draft config's value "
                         "(cold 1,11,20 / warm 1,11,22)")
    a = ap.parse_args()

    src = {}
    with safe_open(os.path.join(a.ckpt, "model.safetensors"), framework="pt") as sf:
        for k in sf.keys():
            src[k] = sf.get_tensor(k)
    print(f"source: {len(src)} tensors from {a.ckpt}")
    scan_finite(src, "source")

    dcfg = json.load(open(os.path.join(a.ckpt, "config.json")))
    tcfg = json.load(open(a.target_config))
    H = int(dcfg.get("hidden_size", tcfg["hidden_size"]))
    n_exp = int(dcfg.get("n_routed_experts", tcfg["n_routed_experts"]))
    qk_nope = int(dcfg.get("qk_nope_head_dim", tcfg["qk_nope_head_dim"]))
    qk_rope = int(dcfg.get("qk_rope_head_dim", tcfg["qk_rope_head_dim"]))
    kv_lora = int(dcfg.get("kv_lora_rank", tcfg["kv_lora_rank"]))

    out = {}

    # 3. fc -> rot_weight (transposed)
    out["model.rot_weight"] = src["fc.weight"].T.contiguous()

    # vocab maps pass through untouched
    for k in ("d2t", "t2d"):
        out[k] = src[k]

    # 4. routed experts: stacked -> per-expert
    w13, w2 = src[TL + "mlp._grouped_w13"], src[TL + "mlp._grouped_w2"]
    I = w13.shape[1] // 2
    for e in range(n_exp):
        out[f"{NEXTN}mlp.experts.{e}.gate_proj.weight"] = w13[e][:I].contiguous()
        out[f"{NEXTN}mlp.experts.{e}.up_proj.weight"] = w13[e][I:].contiguous()
        out[f"{NEXTN}mlp.experts.{e}.down_proj.weight"] = w2[e].contiguous()

    # 5. shared expert: fused up_gate -> gate + up
    ug = src[TL + "mlp.shared_experts.up_gate_proj.weight"]
    J = ug.shape[0] // 2
    out[NEXTN + "mlp.shared_experts.gate_proj.weight"] = ug[:J].contiguous()
    out[NEXTN + "mlp.shared_experts.up_proj.weight"] = ug[J:].contiguous()

    consumed = {"fc.weight", "d2t", "t2d",
                TL + "mlp._grouped_w13", TL + "mlp._grouped_w2",
                TL + "mlp.shared_experts.up_gate_proj.weight"}

    # 1./2. everything else: prefix remap + renames
    for k, v in src.items():
        if k in consumed:
            continue
        if k in HEAD_RENAME:
            out[HEAD_RENAME[k]] = v
            continue
        if not k.startswith(TL):
            raise SystemExit(f"unmapped source key {k!r} -- update the weight map")
        tail = k[len(TL):]
        for old, new in SUBSTR_RENAME.items():
            if tail.startswith(old):
                tail = new + tail[len(old):]
                break
        nk = NEXTN + tail
        # dtype rules (md section 2): bias fp32, scaling bf16
        if nk.endswith("e_score_correction_bias"):
            v = v.float()
        elif nk.endswith("routed_scaling"):
            v = v.to(torch.bfloat16)
        # 6. rope regroup on the two projections that feed q_pe / k_pe
        if nk.endswith("self_attn.q_b_proj.weight"):
            v = rope_regroup(v, qk_nope, qk_nope + qk_rope, qk_nope + qk_rope)
        elif nk.endswith("self_attn.kv_a_proj_with_mqa.weight"):
            v = rope_regroup(v, kv_lora, kv_lora + qk_rope, v.shape[0])
        out[nk] = v.contiguous()

    # ---- guards -----------------------------------------------------------
    leftovers = [k for k in out if k.startswith(("mtp_head.", "draft_model."))]
    if leftovers:
        raise SystemExit(f"trainer prefixes survived: {leftovers[:5]}")
    missing = [k for k in REQUIRED if k not in out]
    if missing:
        raise SystemExit(f"missing required serving keys: {missing}")
    n_expert_keys = sum(1 for k in out if ".experts." in k)
    if n_expert_keys != 3 * n_exp:
        raise SystemExit(f"expert keys {n_expert_keys} != 3*{n_exp}")
    bias = out[NEXTN + "mlp.gate.e_score_correction_bias"]
    if bias.dtype != torch.float32:
        raise SystemExit(f"e_score_correction_bias must stay fp32, got {bias.dtype}")
    if out[NEXTN + "mlp.routed_scaling"].dtype != torch.bfloat16:
        raise SystemExit("routed_scaling must be bf16")
    scan_finite(out, "product")

    # ---- config -----------------------------------------------------------
    cfg = dict(tcfg)
    cfg.update({
        "architectures": ["TargetMTPEagle3ForCausalLM"],
        "model_type": "echo_target",
        "num_hidden_layers": 1,
        # This config describes the DRAFT, not the target. The target
        # (echo1.5) genuinely has 3 MTP depths, but the EAGLE-3 draft is ONE
        # MTP block that sglang runs recursively -- depth comes from the server's
        # --speculative-num-steps, not from stacking blocks. deepseek_nextn.py
        # asserts `num_nextn_predict_layers == 1`, so inheriting the target's 3
        # aborts startup with "Only 1 nextn layer is supported". The training-side
        # draft config does not declare this field at all, hence the hard 1.
        "num_nextn_predict_layers": 1,
        "mtp_reuse_last_layer": False,
        "draft_vocab_size": int(out["d2t"].shape[0]),
        "target_hidden_size": H,
        "num_aux_hidden_states": int(dcfg.get("num_aux_hidden_states", 3)),
        "torch_dtype": "bfloat16",
        "_exported_from": a.ckpt,
    })
    aux = dcfg.get("eagle_aux_hidden_state_layer_ids") or dcfg.get("route_layer_ids")
    if a.aux_layer_ids:
        aux = [int(x) for x in a.aux_layer_ids.split(",")]
    if aux:
        cfg["eagle_aux_hidden_state_layer_ids"] = list(aux)
        # ModelRunner reads the capture layer ids from hf_config.eagle_config
        # (model_runner.py:460); a top-level field alone leaves capture off and
        # the draft then gets a [N, hidden] hidden state instead of the
        # [N, 3*hidden] three-route concat that `rot_weight` expects.
        cfg["eagle_config"] = {
            "use_aux_hidden_state": True,
            "eagle_aux_hidden_state_layer_ids": list(aux),
        }

    os.makedirs(a.out, exist_ok=True)
    save_file(out, os.path.join(a.out, "model.safetensors"), metadata={"format": "pt"})
    json.dump(cfg, open(os.path.join(a.out, "config.json"), "w"), indent=2)
    print(f"exported {len(out)} tensors -> {a.out}")
    print(f"  draft_vocab_size={cfg['draft_vocab_size']} "
          f"aux_ids={cfg.get('eagle_aux_hidden_state_layer_ids')} "
          f"num_aux={cfg['num_aux_hidden_states']}")


if __name__ == "__main__":
    main()
