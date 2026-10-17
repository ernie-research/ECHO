"""Configuration for the target V2 — PyTorch HuggingFace format.

Derived from config.json fields in echo_mtp_step_78000 / moe_mtp_baseline_step_78000.
Registers as model_type="echo_target" so AutoConfig / AutoModel can find it via
trust_remote_code=True from this directory.

Key differences vs DeepSeek-V3:
  - scoring_func = "sigmoid"  (not softmax)
  - use_gated_attn = True     (gate_proj in self_attn)
  - multi_latent_attention    (MLA with q_lora_rank + kv_lora_rank)
  - mtp_reuse_last_layer      (ECHO: reuse backbone final layer as MTP head)
  - routed_scaling_factor_learnable = True
  - vocab_size = 201216
"""
from transformers import PretrainedConfig


class TargetConfig(PretrainedConfig):
    model_type = "echo_target"

    def __init__(
        self,
        # ── core dims ───────────────────────────────────────────────────
        vocab_size: int = 201216,
        hidden_size: int = 1024,
        intermediate_size: int = 5120,      # unused (all MoE), kept for compat
        num_hidden_layers: int = 24,
        num_attention_heads: int = 16,
        num_key_value_heads: int = 16,
        head_dim: int = 128,
        max_position_embeddings: int = 200192,
        # ── MLA (Multi-head Latent Attention) ──────────────────────────
        multi_latent_attention: bool = True,
        q_lora_rank: int = 512,
        kv_lora_rank: int = 512,
        qk_rope_head_dim: int = 64,
        qk_nope_head_dim: int = 192,
        v_head_dim: int = 256,
        use_qk_norm: bool = True,
        qk_norm_type: str = "per_head",
        # ── gated attention ─────────────────────────────────────────────
        use_gated_attn: bool = True,
        use_attn_merger: bool = True,
        # ── RoPE ────────────────────────────────────────────────────────
        rope_theta: float = 10000.0,
        rope_type: str = "rope",
        rotary_dim: int = 64,
        rope_interleave: bool = False,
        # ── MoE ─────────────────────────────────────────────────────────
        moe_intermediate_size: int = 1024,
        moe_latent_size: int = 512,
        n_routed_experts: int = 256,
        num_experts_per_tok: int = 8,
        n_shared_experts: int = 1,
        n_group: int = 1,
        topk_group: int = 1,
        topk_method: str = "noaux_tc",
        scoring_func: str = "sigmoid",
        routed_scaling_factor: float = 2.5,
        routed_scaling_factor_learnable: bool = True,
        first_k_dense_replace: int = 0,
        moe_layer_freq: int = 1,
        moe_router_load_balancing_type: str = "seq_aux_loss",
        moe_shared_expert_gate: bool = False,
        gated_linear_unit: bool = True,
        # ── MTP / ECHO ──────────────────────────────────────────────────
        use_mtp: bool = True,
        num_nextn_predict_layers: int = 1,
        mtp_reuse_last_layer: bool = True,   # True = ECHO, False = DeepSeek/DSpark
        mtp_loss_scaling_factor: float = 0.3,
        use_dense_mtp: bool = False,
        # ── misc ────────────────────────────────────────────────────────
        rms_norm_eps: float = 1e-6,
        tie_word_embeddings: bool = False,
        torch_dtype: str = "bfloat16",
        fp8: str = "e4m3",
        **kwargs,
    ):
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings

        self.multi_latent_attention = multi_latent_attention
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_nope_head_dim = qk_nope_head_dim
        self.v_head_dim = v_head_dim
        self.use_qk_norm = use_qk_norm
        self.qk_norm_type = qk_norm_type

        self.use_gated_attn = use_gated_attn
        self.use_attn_merger = use_attn_merger

        self.rope_theta = rope_theta
        self.rope_type = rope_type
        self.rotary_dim = rotary_dim
        self.rope_interleave = rope_interleave
        self.rope_parameters = None

        self.moe_intermediate_size = moe_intermediate_size
        self.moe_latent_size = moe_latent_size
        self.n_routed_experts = n_routed_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.n_shared_experts = n_shared_experts
        self.n_group = n_group
        self.topk_group = topk_group
        self.topk_method = topk_method
        self.scoring_func = scoring_func
        self.routed_scaling_factor = routed_scaling_factor
        self.routed_scaling_factor_learnable = routed_scaling_factor_learnable
        self.first_k_dense_replace = first_k_dense_replace
        self.moe_layer_freq = moe_layer_freq
        self.moe_router_load_balancing_type = moe_router_load_balancing_type
        self.moe_shared_expert_gate = moe_shared_expert_gate
        self.gated_linear_unit = gated_linear_unit

        self.use_mtp = use_mtp
        self.num_nextn_predict_layers = num_nextn_predict_layers
        self.mtp_reuse_last_layer = mtp_reuse_last_layer
        self.mtp_loss_scaling_factor = mtp_loss_scaling_factor
        self.use_dense_mtp = use_dense_mtp

        self.rms_norm_eps = rms_norm_eps
        self.torch_dtype = torch_dtype
        self.fp8 = fp8
