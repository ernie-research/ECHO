# the target EAGLE-3 draft → sglang 导出映射表（实测反推，2026-09-19）

**这份文档的存在理由**：`export_eagle3_draft_for_sglang.py` 在 2026-09-19 00:52 写入时撞上磁盘
ENOSPC 被截断（181 行 → 46 行，导出逻辑全丢；ECHO 非 git 仓库、全盘无副本）。规则全在代码里、
代码一丢就归零。所以按 SpecStack 的做法（`specforge/export/to_sglang.py` +
`docs/export_weight_map_mla.md`）把映射**文档化**：代码丢了照这张表能重写。

**反推方法**：拿完好的产物 `serve_eagle3_cold/`（00:52，794 张量）当"输出规格"，训练 ckpt
`run4n_eagle3_cold/epoch_0_step_8878`（27 张量）当"输入"，逐张量比 shape + cos + 交叉 cos。
注意产物是 00:52 的权重、训练 ckpt 被 01:02 的保存覆盖过，所以 cos 落在 0.976–1.000
（10 分钟训练漂移），**方向性结论靠交叉 cos ≈ 0 判定，不靠绝对值**。

## 一、张量映射（训练 27 键 → 服务 794 键）

### 1. 直传改名（无变换）

| 训练 key | sglang serving key | 验证 |
|---|---|---|
| `mtp_head.enorm.weight` | `model.enorm.weight` | cos 1.0000 |
| `mtp_head.hnorm.weight` | `model.hnorm.weight` | cos 1.0000 |
| `mtp_head.eh_proj.weight` | `model.eh_proj.weight` | cos 0.9922 |
| `mtp_head.shared_head.norm.weight` | `model.shared_head.norm.weight` | cos 1.0000 |
| `lm_head.weight` | `model.shared_head.head.weight` | cos 0.9846 |
| `mtp_head.transformer_layer.*` | `model.decoder.*`（前缀替换） | — |
| `mtp_head.transformer_layer.mlp.gate.weight` | `model.decoder.mlp.gate.weight` | cos 0.9889 |
| `...mlp.fc1_latent_proj.weight` | `model.decoder.mlp.fc1_latent_proj.weight` | cos 0.9937 |
| `...mlp.fc2_latent_proj.weight` | `model.decoder.mlp.fc2_latent_proj.weight` | cos 0.9939 |
| `...mlp.shared_experts.down_proj.weight` | `model.decoder.mlp.shared_experts.down_proj.weight` | — |
| `...self_attn.{q_a_proj,q_a_layernorm,q_b_proj,kv_a_proj_with_mqa,kv_a_layernorm,kv_b_proj,o_proj}` | 同名，`model.decoder.self_attn.*` | — |
| `{input,post_attention}_layernorm.weight` | 同名，`model.decoder.*` | — |
| `d2t` | `d2t` | **bitEq** |
| `t2d` | `t2d` | **bitEq** |

### 2. 改名 + 语义重命名（容易漏，实测抓到）

| 训练 key | sglang serving key | 说明 |
|---|---|---|
| `...self_attn.gate_proj.weight` (4096,1024) | **`model.decoder.self_attn.attn_out_gate.weight`** | gated attention 的门；名字完全不同，靠 shape+cos 0.9762 定位 |
| `...mlp.gate.routed_scaling_factor_param` (256,) | **`model.decoder.mlp.routed_scaling`** | `models/modeling.py` 用这个名字缩放 top-k；cos 1.0000 |

### 3. 转置

| 训练 key | sglang serving key | 验证 |
|---|---|---|
| `fc.weight` (1024, 3072) | `model.rot_weight` (3072, 1024) = **`fc.weight.T`** | cos(fc.T, rot)=**0.9894**；cos(fc.reshape, rot)=0.0001 |

原因：`DeepseekModelNextN.forward` 在 hnorm 之前做 `matmul(spec_info.hidden_states, rot_weight)`
——正好是 EAGLE-3 三路融合槽位，所以导出转置后不需要改 forward。训练侧 `fc` 是 `nn.Linear`
（`h @ W.T`），故服务侧存 `W.T` 本身。**转置方向搞反不会报错，只会静默降 acceptance。**

### 4. 专家拆分（stacked → per-expert）

sglang 的 FusedMoE loader 自己做 stacking，所以要先拆回 per-expert：

| 训练 key | 拆法 | sglang serving key |
|---|---|---|
| `...mlp._grouped_w13` (256, 2048, 512) | `[e][:1024]` → gate，`[e][1024:]` → up | `model.decoder.mlp.experts.{e}.{gate,up}_proj.weight` (1024,512) |
| `...mlp._grouped_w2` (256, 512, 1024) | `[e]` 直传不转置 | `model.decoder.mlp.experts.{e}.down_proj.weight` (512,1024) |

验证（e=0/7/255）：cos(前半,gate)=0.975–0.980、cos(后半,up)=0.976–0.981，**交叉
cos(前半,up)=0.001/-0.004/0.001** → gate/up 顺序正确、无对调。

### 5. shared expert 拆分

| 训练 key | 拆法 | sglang serving key |
|---|---|---|
| `...mlp.shared_experts.up_gate_proj.weight` (2048,1024) | `[:1024]` → gate，`[1024:]` → up | `model.decoder.mlp.shared_experts.{gate,up}_proj.weight` (1024,1024) |

验证：cos(前半,gate)=0.9837 / cos(后半,up)=0.9839，**交叉 −0.0010**。

### 6. RoPE 偶/奇重组（权重空间置换）

规格来自 `the port log (not published)` §5.2（target 侧同一套，`_rope_regroup_perm` /
`_apply_rope_regroup`）：

- 置换：`[0,2,…,62,1,3,…,63]`（偶数位在前、奇数位在后）
- 作用行段：`self_attn.q_b_proj.weight` **每头**的 rope 段（行 `192:256`，即 qk_nope_head_dim=192
  之后的 qk_rope_head_dim=64 行）、`self_attn.kv_a_proj_with_mqa.weight` 的 k_pe 段（行 `512:576`）
- 根因：the target `_apply_rotary(interleave=False)` 在 split-half 旋转前先
  `cat([x[...,0::2], x[...,1::2]])`（PaddleFleet MLA 偶/奇布局），sglang 的
  `RotaryEmbedding(is_neox_style=True)` 不做这步
- 回退开关：`TARGET_SGLANG_NO_ROPE_REGROUP=1`
- **漏做的后果**：gate-on 真实文本 argmax 一致率 16.4%（做了是 99.61%）。
  §5.3 的教训：gate 把每层注意力贡献压小 3–8×，同样绝对误差被相对放大后才暴露，
  **gate-off 的 99.4% 不能证明数值正确**；诊断要看每层 contribution(delta) 的 cos。

### 7. 不导出的项

- `embed_tokens`：serving 从 target 加载（与 SpecStack `to_sglang.py` 的约定一致，
  产物 794 键里确实没有）。

## 二、dtype 约定（两边不同，别统一）

| 张量 | 训练侧 | 导出侧 | 原因 |
|---|---|---|---|
| `e_score_correction_bias` | **fp32** buffer | **fp32** | bf16 化偏差 ~0.002 加在 O(1e-4..1e-1) 的 sigmoid 分数上 → **layer0 有 59% token 选错专家**（PORT §5.4）。导出脚本必须显式 `.float()` |
| `routed_scaling_factor_param` | **bf16** trainable Parameter | bf16 | fp32 会让 FSDP 报 `Must flatten tensors with uniform dtype`（SUMMARY §4）；它是归一化后的乘性因子，bf16 的 0.4% 相对误差不重排专家 |
| 其余权重 | bf16 | bf16 | — |

## 三、config.json 要点

导出目录的 config 在 target config 基础上改：

- `architectures: ["TargetMTPEagle3ForCausalLM"]`（sglang `EntryClass` 里注册过）
- `num_hidden_layers: 1`（drafter 只有一个 MTP 块）
- `draft_vocab_size: 32000` —— **不设会让 `ParallelLMHead` 造成 201216，加载报 lm_head size mismatch**
- `target_hidden_size: 1024`、`num_aux_hidden_states: 3`、`eagle_aux_hidden_state_layer_ids: [1,11,20]`
  （冷启动；**热启动必须是 `[1,11,22]`**，见 SUMMARY §4 的 route_layer_ids 说明）
- `mtp_reuse_last_layer: false`
- `_exported_from`: 源 ckpt 路径（可追溯）

## 四、导出器必须内置的校验（否则静默出坏 draft）

照 SpecStack `to_sglang.py` 的范式，"weight-name compatibility is **silent when wrong**"：

1. **必需服务键清单**：`model.rot_weight` / `model.shared_head.head.weight` /
   `model.shared_head.norm.weight` / `model.enorm.weight` / `model.hnorm.weight` /
   `model.eh_proj.weight` / `d2t` / `t2d` / 256×3 个 expert 投影 —— 缺一个就抛错，不许静默跳过。
2. **拒绝残留训练前缀**：任何 `mtp_head.` / `draft_model.` 开头的键都是漏改。
3. **isfinite 扫描**：BF16Optimizer 曾缺 non-finite 守卫，一个 NaN 梯度就让全权重永久 NaN 而
   进度条照常显示 loss（SUMMARY §6.6）。导出前扫源 ckpt、导出后扫产物。
   （2026-09-19 01:45 复扫 eagle3_cold 五个档 step_2219/4439/5919/8878/11838：**全部 0 非finite**
   —— §6.6 记的坏档 step_5919 已被 09-19 00:26 的重新保存覆盖，警告已过期，但扫描仍须保留。）
4. **专家计数**：per-expert 键必须恰好 256×3=768 个，加上非 expert 的 26 个 = **794**。

## 五、加载侧验收（`check_sglang_eagle3_load.py`，已通过）

单卡跑 `python -m torch.distributed.run --nproc_per_node 1 check_sglang_eagle3_load.py`，实测：

- registry 解析到 `TargetMTPEagle3ForCausalLM` ✓
- `model.rot_weight` shape (3072,1024)、非零（absmax 0.164）✓
- `get_hot_token_id()` shape (32000,)、严格递增、⊂ [0,201216) ✓
- `lm_head` 32000 行（不是 201216）✓
- `e_score_correction_bias` 全部 fp32 ✓

脚手架注意（省下重新踩坑的时间）：
- `ModelConfig` 之前必须 `register_target()`（transformers 不认 `echo_target`）
- `ParallelLMHead` 读 `get_global_server_args()` → 先
  `set_global_server_args_for_scheduler(ServerArgs(model_path=..., tp_size=1, ...))`
- `DeepseekV2DecoderLayer` 的 communicator 读 dp-attention 全局量 → 先
  `initialize_dp_attention(server_args=..., model_config=...)`
- venv `.venvs/h800-existing-cu13/bin/python` 没有 torchrun，用 `-m torch.distributed.run`
