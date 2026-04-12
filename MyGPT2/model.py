import torch
from torch import nn
import math
import copy

class ModelConfig:
    '''
    This is the model config
    '''
    hidden_size = 768
    num_head = 12
    num_layer = 12
    vocab_size = 50257
    n_positions = 1024
    layer_norm_epsilon=1e-5


class Block(nn.Module):
    '''
    This is the block: attention + FNN
    '''
    def __init__(self):
        super(Block, self).__init__()
        # 1.  LN 1
        self.ln_1 = LayerNorm(ModelConfig.hidden_size)
        # 2. Attention
        self.attn = Attention(is_scale=True)
        # 3. Add &LN2
        self.ln_2 = LayerNorm(ModelConfig.hidden_size)
        # 4. MLP
        self.mlp = MLP()

    def forward(self, x, layer_past=None):
        a, present = self.attn(self.ln_1(x), layer_past=layer_past)
        x = x + a
        m = self.mlp(self.ln_2(x))
        x = x + m
        return x, present


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

class Conv1D(nn.Module):
    def __init__(self, nf, nx):
        super(Conv1D, self).__init__()
        self.nf = nf
        w = torch.empty(nx, nf)
        nn.init.normal_(w, std=0.02)
        self.weight = nn.Parameter(w)
        self.bias = nn.Parameter(torch.zeros(nf))

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        x = x.view(*size_out)
        return x


class Attention(nn.Module):
    '''
    This is the attention class, which containes: proj, matmul, softmat, add
    '''
    def __init__(self, is_scale=False):
        super(Attention, self).__init__()
        self.c_attn = Conv1D(ModelConfig.hidden_size * 3, ModelConfig.hidden_size)
        self.c_proj = Conv1D(ModelConfig.hidden_size, ModelConfig.hidden_size)
        self.scale = is_scale
        n_ctx = ModelConfig.n_positions
        self.register_buffer("bias", torch.tril(torch.ones(n_ctx, n_ctx)).view(1, 1, n_ctx, n_ctx))

        self.n_head = ModelConfig.num_head
        self.split_size = ModelConfig.hidden_size

    def split_heads(self, x, k=False):
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(*new_x_shape) # [batch, seq, head_num, head_dim]
        if k:
            return x.permute(0, 2, 3, 1) # [batch, head_num, head_dim, seq]
        else:
            return x.permute(0, 2, 1, 3) # [batch, head_num, seq, head_dim]

    def merge_heads(self, x):
        # a: [batch, num_heads, seq_len, head_dim]
        x = x.permute(0, 2, 1, 3).contiguous() # [batch, seq_len, num_heads, head_dim]
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)


    def forward(self, x, layer_past = None):
        # 1. proj
        x = self.c_attn(x)
        query, key, value = x.split(self.split_size, dim=2)
        # 2. split heads
        query = self.split_heads(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)

        if layer_past is not None:
            past_key, past_value = layer_past[0].transpose(-2, -1), layer_past[1]  # transpose back cf below
            key = torch.cat((past_key, key), dim=-1)
            value = torch.cat((past_value, value), dim=-2)
        present = torch.stack((key.transpose(-2, -1), value))  # transpose to have same shapes for stacking

        # 3. matmul
        w = torch.matmul(query, key)
        if self.scale:
            w = w / math.sqrt(value.size(-1))
        # mask
        nd, ns = w.size(-2), w.size(-1)
        b = self.bias[:, :, ns-nd:ns, :ns]
        w = w * b - 1e10 * (1 - b)

        # 4. softmax
        w = nn.Softmax(dim=-1)(w)
        a = torch.matmul(w, value)
        # 5. Merge heads
        a = self.merge_heads(a)
        # 6. Proj
        return self.c_proj(a), present


class MLP(nn.Module):
    '''
    Full neural netwotk in every block
    '''
    def __init__(self, ):
        super(MLP, self).__init__()
        self.c_fc = Conv1D(ModelConfig.hidden_size * 4, ModelConfig.hidden_size)
        self.c_proj = Conv1D(ModelConfig.hidden_size, ModelConfig.hidden_size * 4)

    def gelu(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

    def forward(self, x):
        #FFN(x) = W2 · GELU(W1 · x + b1) + b2
        x = self.c_fc(x)
        x = self.gelu(x)
        return self.c_proj(x)

class LMHead(nn.Module):
    '''
    The laste layer: Linear + softmaxt
    '''
    def __init__(self, embd_weights):
        super(LMHead, self).__init__()
        self.set_embeddings_weights(embd_weights)

    def set_embeddings_weights(self, embd_weights):
        weight_shape = embd_weights.shape # [vocab_size, hidden_size]
        self.decoder = nn.Linear(weight_shape[1], weight_shape[0], bias=False) # Linear(in, out), weights(out, in)
        self.decoder.weight = embd_weights


    def forward(self, x):
        lm_logits = self.decoder(x)
        return lm_logits


class GPTModel(nn.Module):
    def __init__(self,):
        super(GPTModel, self).__init__()
        # Token Embed
        self.wte = nn.Embedding(ModelConfig.vocab_size, ModelConfig.hidden_size)
        # Position Embed
        self.wpe = nn.Embedding(ModelConfig.n_positions, ModelConfig.hidden_size)
        block = Block()
        self.h = nn.ModuleList([copy.deepcopy(block) for _ in range(ModelConfig.num_layer)])
        self.ln_f = LayerNorm(ModelConfig.hidden_size, eps=ModelConfig.layer_norm_epsilon)

    def forward(self, input_ids, pos_ids = None, past = None):
        if past is None:
            past_length = 0
            past = [None] * len(self.h)
        else:
            past_length = past[0][0].size(-2)

        if pos_ids is None:
            pos_ids = torch.arange(past_length, input_ids.size(-1) + past_length, dtype=torch.long, device=input_ids.device)
            # 把pos_ids扩展为和input_ids一样
            pos_ids = pos_ids.unsqueeze(0).expand_as(input_ids)
        # Make sure, input_ids: [batch, seq_len]
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_ids.size(-1))
        pos_ids = pos_ids.view(-1, pos_ids.size(-1))

        input_emb = self.wte(input_ids)
        pos_emb = self.wpe(pos_ids)

        hidden_states = input_emb + pos_emb

        presents = []
        for block, layer_past in zip(self.h, past):
            hidden_states, present = block(hidden_states, layer_past)
            presents.append(present)

        hidden_states = self.ln_f(hidden_states)
        output_shape = input_shape + (hidden_states.size(-1), )

        return hidden_states.view(*output_shape), presents


class GPT2LMHeadModel(nn.Module):
    def __init__(self):
        super(GPT2LMHeadModel, self).__init__()
        self.transformer = GPTModel()
        self.lm_head = LMHead(self.transformer.wte.weight)

    def set_tied(self):
        """ Make sure we are sharing the embeddings
        """
        self.lm_head.set_embeddings_weights(self.transformer.wte.weight)

    def forward(self, input_ids, pos_ids=None, past=None):
        hidden_states, presents = self.transformer(input_ids, pos_ids, past)
        lm_logits = self.lm_head(hidden_states)
        return lm_logits, presents
