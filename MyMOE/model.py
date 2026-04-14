import torch
from torch import nn
from torch.functional import F

class ModelConfig:
    topk = 4
    num_expert = 32


class Expert(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.activate = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.activate(x)
        x = self.fc2(x)
        return x
    

class TopKGate(nn.Module):
    def __init__(self, input_dim, k, num_expert):
        super().__init__()
        self.k = k
        self.num_expert = num_expert
        self.gate_network = nn.Linear(input_dim, k)
    

    def forward(self, x):
        gate_logits = self.gate_network(x) # [N, num_exprts]
        gate_weights = F.softmax(gate_logits, dim=-1)
        top_k_weights, top_k_indices = torch.topk(gate_weights, self.k, dim=-1)
        # 归一化
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)

        return gate_weights, top_k_weights, top_k_indices


class MoELayer(nn.Module):
    def __init__(self, input_dim, output_dim, expert_hidden_dim = None):
        super().__init__()
        
        self.gate = TopKGate(ModelConfig.topk, ModelConfig.num_expert)
        self.output_dim = output_dim
        if expert_hidden_dim == None:
            self.expert_hidden_dim = 4 * input_dim

        self.experts = nn.ModuleList([Expert(input_dim, self.expert_hidden_dim, self.output_dim) for _ in range(ModelConfig.num_expert)])

    def forward(self, x):
        origin_shape = x.shape
        # x: [batch, seq_len, input_dim]
        # 1. 需要找到每个token需要用哪几个专家，每个专家的打分;
        ## 1.1 可以变成2维，语义更直接。
        x_flat = x.view(-1, x.size(-1)) # [N, input_dim], N = batch * seq_len
        ## 1.2 每个token, 在各个专家上的打分，以及top_k 专家对应的下标。
        gate_weights, top_k_indices = self.gate(x_flat) # full_weights: [N, num_expert], [N, k]

        # flaten indices for torch.where
        flat_top_k_indices = top_k_indices.view(-1)

        x_flat = x_flat.repeat_interleave(ModelConfig.topk, dim=0) #这里算一个小技巧

        # 把token 分配给每个专家去计算
        expert_outputs = []
        for i in range (ModelConfig.num_expert):
            # 找到了这个专家对应的tokens idx. 但是这里的idx还需要再转换一下，因为被flat了。
            idx = torch.where(i == flat_top_k_indices)[0]
            if idx.numel() > 0:
                expert_input = x_flat[idx]
                expert_output = self.experts[i](expert_input)
                # 每个token在当前expert的输出
                expert_outputs.append((idx, expert_output))
        

        final_output = torch.zeros(x_flat.size(0), self.output_dim, device=x.device)
        # 现在我已经拿到了每个token在k个专家上的输出。现在需要组合每个专家的输出进行加权
        for idx, output in expert_outputs:
            # which token
            origin_idx = idx // ModelConfig.topk
            # which expert
            expert_idx = flat_top_k_indices[idx]

            weight = gate_weights[origin_idx, expert_idx].unsquueze(1)

            weighted_output = weight * output

            # 使用 index_add_ 进行分散-累加操作
            final_output.index_add_(0, origin_idx, weighted_output)
        # [batch, seq_len, output_dim]    
        final_output = final_output.view(origin_shape[0], origin_shape[1], self.output_dim)

        

class TransformerBlockWithMoE(nn.Module):
    
    def __init__(self, embed_dim, num_heads):
        super().__init__()

        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)

        self.ln1 = nn.LayerNorm(embed_dim)

        self.ln2 = nn.LayerNorm(embed_dim)

        self.moe = MoELayer(embed_dim, embed_dim)

        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        attn = self.attention(x, x, x)
        # add & ln
        x = self.ln1(x + self.dropout(attn))

        output, gated_weights = self.moe(x)

        x = x + self.dropout(output)

        x = self.ln2(x)
        return x, gated_weights



        


        
