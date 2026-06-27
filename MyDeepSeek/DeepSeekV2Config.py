from typing import Optional, Tuple

from torch import nn
import torch

class DeepseekV2Config:
    num_hidden_layers =  60 # Transformer层的数量
    hidden_size = 5120 # 隐藏层的大小
    num_attention_heads = 128 # 注意力头的数量
    kv_lora_rank = 512 # KV压缩维度
    q_lora_rank = 1536 # Query压缩维度
    qk_rope_head_dim = 64 # 解耦Query和Key的每个头部维度
    n_shared_experts = 2 # MoE层中的共享专家数量
    n_routed_experts= 160 # MoE层中的路由专家数量
    moe_intermediate_size= 1536 # 每个MoE专家的中间隐藏层的维度
    num_experts_per_tok= 6 # 每个token激活的专家数量
    routed_scaling_factor= 16.0 # 路由专家的缩放因子
    rms_norm_eps= 1e-06 # RMS归一化的epsilon值

class DeepseekV2Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""
    def __init__(self, config: DeepseekV2Config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        # 对应 query 压缩后的隐向量的维度 d'_c
        self.q_lora_rank = config.q_lora_rank
        # query和key的隐藏向量中，应用rope部分的维度，对应d_h^R
        self.qk_rope_head_dim = config.qk_rope_head_dim
        # 对应 key-value 压缩后的隐向量维度 d_c
        self.kv_lora_rank = config.kv_lora_rank
        # value 的一个注意力头的隐藏层维度
        self.v_head_dim = config.v_head_dim
        # 向量中不应用rope部分的维度
        self.qk_nope_head_dim = config.qk_nope_head_dim
        # 每一个注意力头的维度应该是nope和rope两部分之和
        self.q_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
        self.is_causal = True

        # MLA 中对 Q 投影矩阵也做了一个低秩分解，对应生成 q_a_proj 和 q_b_proj 两个矩阵，即两阶段投影：先将hidden_size投影到q_lora_rank，再投影到最终维度
        # 对query进行压缩，即down-projection。即，第一阶段投影：hidden_size -> q_lora_rank，对应论文公式中的W^DQ
        self.q_a_proj = nn.Linear(
            self.hidden_size, config.q_lora_rank, bias=config.attention_bias
        )
        self.q_a_layernorm = DeepseekV2RMSNorm(config.q_lora_rank)
        # 对压缩后的query映射成高维，即up-projection。对应上述公式中的W^UQ和W^QR合并后的大矩阵，仅仅只是内存放在一起。
        # q_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim = 128 + 64
        self.q_b_proj = nn.Linear(
            config.q_lora_rank, self.num_heads * self.q_head_dim, bias=False
        )

        # KV向量的生成也是先投影到一个低维的 compressed_kv 向量（对应c_t^{KV}），再升维展开
        # 对应论文公式中的W^{DKV}和W^{KR}
        self.kv_a_proj_with_mqa = nn.Linear(
            self.hidden_size,
            config.kv_lora_rank + config.qk_rope_head_dim,
            bias=config.attention_bias,
        )
        self.kv_a_layernorm = DeepseekV2RMSNorm(config.kv_lora_rank)
        # 对应论文公式中的W^{UK}和W^{UV}，由于 W^{UK} 只涉及 non-rope 的部分，所以维度中把 qk_rope_head_dim 去掉了
        self.kv_b_proj = nn.Linear(
            config.kv_lora_rank,
            self.num_heads
            * (self.q_head_dim - self.qk_rope_head_dim + self.v_head_dim),
            bias=False,
        )

        # 对应论文公式的第 47 行
        self.o_proj = nn.Linear(
            self.num_heads * self.v_head_dim,
            self.hidden_size,
            bias=config.attention_bias,
        )
        self._init_rope()

        self.softmax_scale = self.q_head_dim ** (-0.5)
        if self.config.rope_scaling is not None:
            mscale_all_dim = self.config.rope_scaling.get("mscale_all_dim", 0)
            scaling_factor = self.config.rope_scaling["factor"]
            if mscale_all_dim:
                mscale = yarn_get_mscale(scaling_factor, mscale_all_dim)
                self.softmax_scale = self.softmax_scale * mscale * mscale

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None, # V2代码中，kv cache存储的是全部缓存，不是压缩后的
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        # hidden_states对应公式中的h_t，的shape是(batch_size, seq_length,hidden_size)
        bsz, q_len, _ = hidden_states.size()

        q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
        q_nope, q_pe = torch.split(
            q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        compressed_kv, k_pe = torch.split(
            compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )
        k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)
        kv = (
            self.kv_b_proj(self.kv_a_layernorm(compressed_kv))
            .view(bsz, q_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
            .transpose(1, 2)
        )

        k_nope, value_states = torch.split(
            kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1
        )
        kv_seq_len = value_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)

        q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe, cos, sin, position_ids)

        query_states = k_pe.new_empty(bsz, self.num_heads, q_len, self.q_head_dim)
        query_states[:, :, :, : self.qk_nope_head_dim] = q_nope
        query_states[:, :, :, self.qk_nope_head_dim :] = q_pe

        key_states = k_pe.new_empty(bsz, self.num_heads, q_len, self.q_head_dim)
        key_states[:, :, :, : self.qk_nope_head_dim] = k_nope
        key_states[:, :, :, self.qk_nope_head_dim :] = k_pe
        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        attn_weights = (
            torch.matmul(query_states, key_states.transpose(2, 3)) * self.softmax_scale
        )

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.attention_dropout, training=self.training
        )
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.v_head_dim)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

