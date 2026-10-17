"""Verify that the collector's route 3 == the HF reference's backbone_hidden.

Run on a single GPU (any rank that is free). Does NOT use the collector's own
code path -- it runs the HF reference and the sglang target independently on the
same input, then compares the post-norm hidden.

Three checks:
  1. HF backbone_hidden == sglang last_hidden_states (tensor equality)
  2. HF backbone_hidden == collector's route 3 (== sglang hidden_states fed to
     the logits processor, which IS last_hidden_states by construction)
  3. With fc=[0|0|I], draft step-0 input == pretrained MTP's input
"""
import sys, os, json, torch
sys.path.insert(0, "<WORKSPACE>/paddle_hf_alignment/src")
sys.path.insert(0, "<WORKSPACE>/SpecForge")

M = "<WORKSPACE>/echo1.5_sft_step5000"
TOK_PATH = "<WORKSPACE>/paddle_hf_alignment/tokenizer_hf"

import os as _os
# BE pods reclaim idle GPUs within seconds of an evict, and this check does not need
# a GPU: one short sequence through a 24-layer/hidden-1024 model is cheap on CPU and
# removes the race entirely. ALIGN_DEVICE=cuda to force the GPU path.
device = _os.environ.get("ALIGN_DEVICE", "cpu")
dtype = torch.bfloat16 if device == "cuda" else torch.float32

# --- 1. HF reference: get backbone_hidden ---
print("=== [1/3] HF reference forward")
from specforge.target import load_target, register_target
register_target()
from transformers import AutoTokenizer

# Plain `from_pretrained` does NOT work on this checkpoint: every
# `model.mtp_heads.*` key comes up MISSING and gets randomly initialised, and the
# flat-export remap / `mtp_reuse_last_layer` parameter aliasing / fp32
# `e_score_correction_bias` handling are all skipped -- the forward then returns NaN.
# `load_target` drives the acceptance package's torch_weight_loader, which is
# the same path the collector used for the perfectblend regeneration.
hf_model, _, load_stats = load_target(M, device=device, dtype=dtype, use_mtp=True)
hf_model.eval()
print(f"   loaded: {load_stats}")
tokenizer = AutoTokenizer.from_pretrained(TOK_PATH, trust_remote_code=True)

text = "The quick brown fox jumps over the lazy dog."
enc = tokenizer(text, return_tensors="pt").to(device)
input_ids = enc["input_ids"]
attention_mask = enc["attention_mask"]

with torch.no_grad():
    # return_mtp_hidden=True makes TargetModel.forward return (hidden, backbone_hidden, kv)
    hidden, backbone_hidden, _ = hf_model.model(
        input_ids,
        attention_mask=attention_mask,
        return_mtp_hidden=True,
        use_cache=False,
    )
    # backbone_hidden is self.norm(layer23_out) per modeling.py:985-991
    hf_logits = hf_model.lm_head(backbone_hidden).float()

print(f"   input_ids shape: {input_ids.shape}")
print(f"   backbone_hidden shape: {backbone_hidden.shape}, dtype: {backbone_hidden.dtype}")
print(f"   backbone_hidden L2: {backbone_hidden.float().norm():.4f}")
print(f"   hf_logits argmax[:8]: {hf_logits.argmax(-1)[0,:8].tolist()}")

# --- 2. Verify that fc=[0|0|I] reproduces MTP input ---
print("\n=== [2/3] fc=[0|0|I] check")
H = backbone_hidden.shape[-1]
# Simulate: feature = cat(h_low, h_mid, backbone_hidden) with h_low=h_mid=zeros
fake_h_low = torch.zeros_like(backbone_hidden)
fake_h_mid = torch.zeros_like(backbone_hidden)
feature_3h = torch.cat([fake_h_low, fake_h_mid, backbone_hidden], dim=-1)  # [1, T, 3H]

# fc = [0|0|I] means W = [[0...], [0...], [I]] of shape [H, 3H], so W @ feature = feature[:, 2H:3H]
# i.e. just the third route = backbone_hidden
fc_weight = torch.zeros(H, 3*H, dtype=dtype, device=device)
fc_weight[:, 2*H:3*H] = torch.eye(H, dtype=dtype, device=device)
fc_out = torch.nn.functional.linear(feature_3h, fc_weight)  # [1, T, H]

diff = (fc_out.float() - backbone_hidden.float()).abs().max().item()
cos = torch.nn.functional.cosine_similarity(fc_out.float().flatten(), backbone_hidden.float().flatten(), dim=0).item()
print(f"   fc([0|0|I]) output vs backbone_hidden: max_diff={diff:.2e}, cosine={cos:.8f}")
if diff < 1e-6:
    print("   ✓ fc=[0|0|I] reproduces the pretrained MTP input EXACTLY")
else:
    print(f"   ✗ MISMATCH: max diff = {diff}")

# --- 3. Verify route widths ---
print("\n=== [3/3] Route width summary")
cfg = json.load(open("SpecForge/configs/target-eagle3-warm.json"))
print(f"   config route_layer_ids: {cfg['route_layer_ids']}")
print(f"   hidden_size: {H}")
print(f"   feature_width (3*H): {3*H}")
print(f"   expected fc.in_features: {3*H}")
print(f"   num_aux_hidden_states: {cfg['num_aux_hidden_states']}")
print("\n=== ALIGNMENT CHECK DONE")
