"""Offline AR_1 for the trained the target EAGLE-3 draft, under both input alignments.

Why: serving reports accept_len == 1.0000 exactly (zero accepted drafts) for every
EAGLE-3 checkpoint, while the weights, aux capture, RoPE regroup and d2t map were
all verified fine. The remaining suspect is the embed/hidden pairing:

  training  (specforge/core/eagle3.py:492 + SdpaLikeAdapter.step_view idx=0)
            draft(embed(tok_i), h_i)      -> target's distribution at i  (= tok_{i+1})
            i.e. input_ids are NOT shifted

  serving   (sglang eagle_worker_v2._draft_extend_for_prefill)
            batch.input_ids[pt:pt+n] = cat(input_ids[1:], next_token)
            draft(embed(tok_{i+1}), h_i)  -> tok_{i+2}
            i.e. input_ids ARE shifted left by one

This script runs the trained draft both ways on a real reference sequence and
reports the first-step acceptance AR_1 for each. A high AR_1 for "unshifted" and
~0 for "shifted" pins the mismatch on the alignment, not on training quality.

Runs on one GPU; the target is loaded with no MTP.
"""

import argparse
import json
import os
import sys

import torch

ACC = "<WORKSPACE>/paddle_hf_alignment"
sys.path.insert(0, os.path.join(ACC, "src"))
sys.path.insert(0, "<WORKSPACE>/SpecForge")


def build_target(path, device, dtype):
    from modeling.torch_weight_loader import build_and_load_torch

    model, config, _ = build_and_load_torch(
        path, device=device, dtype=dtype, no_mtp=True, preserve_mtp_sharing=False
    )
    model.model.prepare_inference_weights()
    model.eval().requires_grad_(False)
    return model, config


def capture_aux(model, input_ids, layer_ids):
    """Return (aux concat [1,T,3H], last hidden [1,T,H], logits [1,T,V])."""
    grabbed = {}
    handles = []
    layers = model.model.layers

    def hook(idx):
        def fn(module, args, output):
            grabbed[idx] = output[0] if isinstance(output, tuple) else output

        return fn

    for i in layer_ids:
        handles.append(layers[i].register_forward_hook(hook(i)))
    try:
        out = model(input_ids=input_ids, use_cache=False)
    finally:
        for h in handles:
            h.remove()
    aux = torch.cat([grabbed[i] for i in layer_ids], dim=-1)
    return aux, out.logits


def build_draft(ckpt, device, dtype):
    from specforge.target import register_target
    from specforge.modeling.draft.target_mtp_eagle import (
        TargetMTPEagleConfig,
        TargetMTPEagleModel,
    )
    from safetensors.torch import load_file

    register_target()
    cfg = TargetMTPEagleConfig(**json.load(open(os.path.join(ckpt, "config.json"))))
    model = TargetMTPEagleModel(cfg)
    state = load_file(os.path.join(ckpt, "model.safetensors"))
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"  draft load: missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        print(f"    first missing: {missing[:5]}")
    if unexpected:
        print(f"    first unexpected: {unexpected[:5]}")
    model.to(device=device, dtype=dtype).eval().requires_grad_(False)
    return model, cfg, state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="<WORKSPACE>/echo1.5_sft_step5000")
    ap.add_argument("--draft", default="<WORKSPACE>/run4n_eagle3_cold/epoch_1_step_59190")
    ap.add_argument("--greedy", default="<WORKSPACE>/logs/sglang/greedy_nomtp.json")
    ap.add_argument("--num-prompts", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    dtype = torch.bfloat16
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        os.path.join(ACC, "tokenizer_hf"), trust_remote_code=True
    )
    print("loading target ...")
    target, tcfg = build_target(args.target, args.device, dtype)
    print("loading draft ...")
    draft, dcfg, state = build_draft(args.draft, args.device, dtype)

    # The draft checkpoint deliberately omits embed_tokens (serving binds the
    # target's). Without this the probe would run on a random embedding.
    with torch.no_grad():
        draft.embed_tokens.weight.copy_(
            target.model.embed_tokens.weight.to(draft.embed_tokens.weight.dtype)
        )
    print("  bound target embed_tokens into draft")

    route = list(getattr(dcfg, "route_layer_ids", None) or [1, 11, 20])
    print(f"  route_layer_ids={route}  draft_vocab_size={dcfg.draft_vocab_size}")
    d2t = state["d2t"].to(args.device)
    hot = d2t + torch.arange(d2t.shape[0], device=args.device)

    rows = json.load(open(args.greedy))[: args.num_prompts]
    stats = {"unshifted": [0, 0], "shifted": [0, 0]}

    for r in rows:
        prompt_ids = tok(r["prompt"], return_tensors="pt").input_ids.to(args.device)
        ids = torch.cat(
            [prompt_ids, torch.tensor([r["output_ids"]], device=args.device)], dim=1
        )
        T = ids.shape[1]
        aux, logits = capture_aux(target, ids, route)
        h = draft.project_hidden_states(aux.to(dtype))
        pos = torch.arange(T, device=args.device).unsqueeze(0)
        mask = torch.triu(
            torch.full((1, 1, T, T), float("-inf"), device=args.device, dtype=dtype),
            diagonal=1,
        )
        gen_start = prompt_ids.shape[1]

        for name in ("unshifted", "shifted"):
            # unshifted: emb(tok_i) with h_i, predicting tok_{i+1}   (training)
            # shifted:   emb(tok_{i+1}) with h_i, predicting tok_{i+2} (sglang)
            src = ids if name == "unshifted" else torch.cat(
                [ids[:, 1:], ids[:, -1:]], dim=1
            )
            emb = draft.embed_input_ids(src).to(dtype)
            out = draft.backbone(
                input_embeds=emb,
                hidden_states=h,
                cache_hidden=None,
                attention_mask=mask,
                position_ids=pos,
            )
            draft_logits = draft.compute_logits(out)
            pred_draft = draft_logits.argmax(-1)[0]
            pred = hot[pred_draft.clamp(max=hot.shape[0] - 1)]
            off = 1 if name == "unshifted" else 2
            lo, hi = gen_start, T - off
            if hi <= lo:
                continue
            gold = ids[0, lo + off : hi + off]
            got = pred[lo:hi]
            stats[name][0] += int((got == gold).sum())
            stats[name][1] += int(gold.numel())

    print("\n  alignment  |  AR_1 (first-step acceptance)")
    for name, (c, n) in stats.items():
        print(f"  {name:10s} |  {c}/{n} = {c / n:.4f}" if n else f"  {name}: no data")
    tag = "training(unshifted)" if stats["unshifted"][0] > stats["shifted"][0] else "sglang(shifted)"
    print(f"\n  better alignment: {tag}")


if __name__ == "__main__":
    main()

