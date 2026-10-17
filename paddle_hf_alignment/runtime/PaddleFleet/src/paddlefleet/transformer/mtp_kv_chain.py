# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""MTP cross-depth K/V chain attention (mtp_kv_cache_mode="chain").

Eagle's diagonal kernel, ported to Paddle. The MTP layers run as a for-loop
over depths (all T positions in parallel). Each depth predicts one shifted
token per position, so cross-depth attention is:

  - depth 0: full T x T causal self-attention over its own K0/V0 (the anchor).
  - depth d >= 1: each query position t attends to
      * the anchor causal context K0[0..t]   (full T x T block), AND
      * one NEW draft key per prior depth at the SAME position t:
            K_i[t] for i in 1..d            (one diagonal column per depth)
    plus its own diagonal contribution.

This is exactly "append one new draft K/V per depth per position", computed in
parallel over the T positions. It materializes only [T, T + d] scores per
depth (anchor T x T + d diagonal columns), so there is no [T, (d+1)*T] blow-up
and no OOM. The fused flashmask kernel cannot express this (its causal mode
pins the T queries to the FRONT of a longer key axis and offers no per-depth
query offset), which is why eagle hand-rolled this matmul; we mirror it.

SWA over the chain = keep at most `window` of the most-recent diagonal columns
(i.e. limit how many prior depths a query may look back across). window=None
keeps all prior depths.

Reference (torch): eagle/traineagle3/cnets.py:288-316.
Paddle numerical spec: tests/single_card_tests/transformer/
test_mtp_kv_chain_reference.py.
"""

from __future__ import annotations

import numpy as np
import paddle


class MTPKVChain:
    """Rank-local cross-depth diagonal K/V chain.

    Stores per-depth post-RoPE K/V (each [B, n_kv, T, dh]) in depth order. One
    instance is attached to all MTP layers' attention on a PP rank; the first
    depth resets it each forward.

    NOT safe under full activation recompute or cross-PP-stage placement; the
    caller asserts those out before use.
    """

    def __init__(self):
        self._k = []
        self._v = []

    def reset(self):
        self._k = []
        self._v = []

    def append(self, k: paddle.Tensor, v: paddle.Tensor):
        self._k.append(k)
        self._v.append(v)

    @property
    def depth_filled(self):
        return len(self._k)

    def k_list(self):
        return self._k

    def v_list(self):
        return self._v


def _repeat_kv(x: paddle.Tensor, n_rep: int) -> paddle.Tensor:
    """[B, n_kv, T, dh] -> [B, n_kv*n_rep, T, dh]."""
    if n_rep == 1:
        return x
    b, n_kv, t, dh = x.shape
    x = x.unsqueeze(2).expand([b, n_kv, n_rep, t, dh])
    return x.reshape([b, n_kv * n_rep, t, dh])


def mtp_chain_attention(
    query: paddle.Tensor,  # [B, T, n_heads, dh], post-RoPE
    key: paddle.Tensor,  # [B, T, n_kv, dh], post-RoPE (current depth)
    value: paddle.Tensor,  # [B, T, n_kv, dh] (current depth)
    chain: MTPKVChain,
    *,
    n_rep: int,
    softmax_scale: float | None = None,
    window: int | None = None,
    anchor_only: bool = False,
) -> paddle.Tensor:
    """Eagle diagonal attention for one MTP depth.

    Appends (key, value) to the chain, then:
      - depth 0 (chain had nothing before this append): full T x T causal.
      - depth d >= 1: anchor T x T causal + one diagonal column per prior depth
        in the window, + own diagonal.

    Modes:
      - anchor_only=False (default, EAGLE "chain"): include prior-depth diagonal
        columns (subject to `window`). Cross-depth draft K/V flows.
      - anchor_only=True (GLM-5.2 "anchor"/KVShare): drop ALL diagonal columns;
        every depth attends ONLY to depth-0's anchor K0/V0 (target-derived).
        No attention to prior draft depths. (Equivalent to window=0.)

    Returns [B, T, n_heads * dh].

    Shapes inside: transpose to [B, n_heads, T, dh] for the matmuls.
    """
    # To [B, heads, T, dh].
    q = query.transpose([0, 2, 1, 3])
    k = key.transpose([0, 2, 1, 3])
    v = value.transpose([0, 2, 1, 3])

    chain.append(k, v)
    cache_k = chain.k_list()
    cache_v = chain.v_list()
    lck = len(cache_k)  # number of depths so far (including current)

    if softmax_scale is None:
        dh = q.shape[-1]
        softmax_scale = 1.0 / np.sqrt(dh)

    # GQA expand each block.
    cache_k_r = [_repeat_kv(ki, n_rep) for ki in cache_k]
    cache_v_r = [_repeat_kv(vi, n_rep) for vi in cache_v]

    B, Hq, T, dh = q.shape
    neg_inf = float(np.finfo(np.float32).min)

    k0 = cache_k_r[0]
    v0 = cache_v_r[0]

    # Anchor: full T x T causal scores.
    attn_weights = paddle.matmul(q, k0.transpose([0, 1, 3, 2])) * softmax_scale
    q_idx = paddle.arange(T).reshape([T, 1])
    k_idx = paddle.arange(T).reshape([1, T])
    causal = k_idx <= q_idx
    causal_mask = paddle.where(
        causal,
        paddle.zeros([T, T], dtype=attn_weights.dtype),
        paddle.full([T, T], neg_inf, dtype=attn_weights.dtype),
    )
    attn_weights = attn_weights + causal_mask.unsqueeze([0, 1])

    # Prior-depth diagonal columns.
    #   anchor_only (GLM-5.2): NO diagonals -> attend to anchor K0/V0 ONLY.
    #   chain (EAGLE): one diagonal column per prior depth, limited by `window`
    #     (window = max number of most-recent prior depths kept; None = all).
    if anchor_only:
        diag_indices = []
    elif window is None:
        diag_indices = list(range(1, lck))
    else:
        first_diag = max(1, lck - window)
        diag_indices = list(range(first_diag, lck))

    if diag_indices:
        extras = []
        for i in diag_indices:
            ki = cache_k_r[i]  # [B, Hq, T, dh]
            diag_w = (q * ki).sum(-1) * softmax_scale  # [B, Hq, T]
            extras.append(diag_w.unsqueeze(-1))
        attn_weights = paddle.concat(
            [attn_weights] + extras, axis=-1
        )  # [B, Hq, T, T + len(diag_indices)]

    attn_weights = paddle.nn.functional.softmax(
        attn_weights.astype("float32"), axis=-1
    ).astype(q.dtype)

    # Aggregate values: anchor block then diagonal columns.
    probs0 = attn_weights[..., :T]
    attn_output = paddle.matmul(probs0, v0)  # [B, Hq, T, dh]
    for col, i in enumerate(diag_indices):
        vi = cache_v_r[i]  # [B, Hq, T, dh]
        probs_i = attn_weights[..., T + col]  # [B, Hq, T]
        attn_output = attn_output + probs_i.unsqueeze(-1) * vi

    attn_output = attn_output.transpose([0, 2, 1, 3]).reshape([B, T, Hq * dh])
    return attn_output
