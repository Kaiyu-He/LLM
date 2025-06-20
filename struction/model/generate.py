import time
from typing import Optional

import torch
from torch import nn, Tensor
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(
            self,
            d_model: int = 512,
            dim_k: int = 64,
            dim_v: int = 64,
            n_head: int = 8,
            n_kv_head: int = 8,
            max_len: int = 1024,
            device: str = "cuda",
    ):
        super().__init__()

        self.d_model = d_model
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.n_kv_head = n_kv_head
        self.n_head = n_head

        self.W_q = nn.Linear(d_model, n_head * dim_k, bias=False)
        self.W_k = nn.Linear(d_model, n_kv_head * dim_k, bias=False)
        self.W_v = nn.Linear(d_model, n_kv_head * dim_v, bias=False)
        self.W_o = nn.Linear(n_head * dim_v, d_model, bias=False)

    def forward(
            self,
            inputs: Tensor,
            return_kvs: bool = False,
            kv_cache: Tensor = None,
            use_flash_attn: bool = False,
            is_causal: bool = False,
            mask: torch.Tensor = None,
    ):
        """
        inputs: (B T D)
        ---
        B: batch size
        T: sequence length
        D: model dimension
        H: number of kv head
        """
        B, T, D = inputs.shape
        if kv_cache is not None:
            inputs = inputs[:, -1:, :]

        # 分头
        q = self.W_q(inputs).view(B, -1, self.n_head, self.dim_k)  # (B, T, H, dim_k)
        k = self.W_k(inputs).view(B, -1, self.n_kv_head, self.dim_k)  # (B, T, H, dim_k)
        v = self.W_v(inputs).view(B, -1, self.n_kv_head, self.dim_v)  # (B, T, H, dim_v)

        if kv_cache is not None:
            assert isinstance(kv_cache, tuple)
            k_cache, v_cache = kv_cache
            k = torch.cat((k_cache, k), dim=1)  # (B, T_ctx + 1, H, dim_k)
            v = torch.cat((v_cache, v), dim=1)  # (B, T_ctx + 1, H, dim_v)
        if return_kvs:
            kvs = (k, v)
        else:
            kvs = None
        q, k, v = map(lambda e: e.transpose(1, 2), (q, k, v))  # (B, H, T, dim_k)

        if use_flash_attn:
            output = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)  # (B, H, T, dim_v)
        else:
            scores = torch.matmul(q, k.transpose(-2, -1)) / (self.dim_k ** 0.5)  # (B, H, T, T)
            if mask is not None:
                scores = scores.masked_fill(mask == 0, float('-inf'))
            attn = F.softmax(scores, dim=-1)  # (B, H, T, T)
            output = torch.matmul(attn, v)  # (B, H, T, dim_v)

        output = output.transpose(1, 2).contiguous().view(B, -1, self.d_model)  # (B, T, D)
        output = self.W_o(output)  # (B, T, D)
        return {
            "output": output,
            "kvs": kvs,
        }


class FFN(nn.Module):
    def __init__(
            self,
            d_model: int = 512,
            d_ffn: int = 128,
    ):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ffn)
        self.fc2 = nn.Linear(d_ffn, d_model)
        self.gelu = nn.GELU()

    def forward(self, x):
        return self.fc2(self.gelu(self.fc1(x)))


class GPT2Block(nn.Module):
    def __init__(
            self,
            d_model=512,
            dim_k: int = 64,
            dim_v: int = 64,
            n_head: int = 8,
            n_kv_head: int = 8,
            d_ffn=3072,
            dropout=0.1,
            max_len: int = 1024,
            device: str = "cuda",
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)  # 前置 LayerNorm
        self.attn = MultiHeadAttention(
            d_model=d_model,
            dim_k=dim_k,
            dim_v=dim_v,
            n_head=n_head,
            n_kv_head=n_kv_head,
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FFN(
            d_model=d_model,
            d_ffn=d_ffn
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
            self,
            x,
            mask=None,
            return_kvs: bool = False,
            kv_cache: Tensor = None,
    ):
        # 自注意力 + 残差
        attn = self.attn(
            self.ln1(x),
            return_kvs=return_kvs,
            kv_cache=kv_cache,
            mask=mask,
        )
        if return_kvs:
            kvs = attn['kvs']
        else:
            kvs = None
        attn = attn['output']
        x = x + self.dropout(attn)
        # FFN + 残差
        ffn_output = self.ffn(self.ln2(x))
        x = x + self.dropout(ffn_output)
        return {
            "output": x,
            "kvs": kvs
        }


class GPT2Config:
    def __init__(
            self,
            vocab_size=50257,  # GPT-2 的词汇表大小
            d_model=512,  # 隐藏层维度
            dim_k: int = 64,
            dim_v: int = 64,
            n_layer=8,  # Transformer 层数
            n_head=8,  # 注意力头数
            n_kv_head=8,
            d_ffn=3072,  # FFN 中间层维度
            dropout=0.1,  # Dropout 率
            max_len: int = 1024,
            device: str = "cuda",

    ):
        self.vocab_size = vocab_size
        self.n_positions = max_len
        self.d_model = d_model
        self.n_layer = n_layer
        self.n_head = n_head
        self.d_ffn = d_ffn
        self.dropout = dropout
        self.n_kv_head = n_kv_head
        self.max_len = max_len
        self.device = device
        self.dim_k = dim_k
        self.dim_v = dim_v


class GPT2Model(nn.Module):
    def __init__(
            self,
            config: GPT2Config
    ):
        super().__init__()
        self.config = config

        # 输入嵌入层
        self.wte = nn.Embedding(config.vocab_size, config.d_model)
        self.wpe = nn.Embedding(config.n_positions, config.d_model)
        self.drop = nn.Dropout(config.dropout)

        # Transformer 层
        self.blocks = nn.ModuleList([
            GPT2Block(
                d_model=config.d_model,
                dim_k=config.dim_k,
                dim_v=config.dim_v,
                n_head=config.n_head,
                n_kv_head=config.n_kv_head,
                d_ffn=config.d_ffn,
                dropout=config.dropout,
            ) for _ in range(config.n_layer)
        ])

        # 输出层
        self.norm = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # 权重绑定（输入嵌入与输出层共享权重）
        self.head.weight = self.wte.weight

    def forward(
            self,
            input_ids,
            mask=None,
            return_kvs: bool = False,
            kv_cache: Tensor = None,
    ):
        device = input_ids.device
        B, T = input_ids.shape

        # 生成位置ID
        pos_ids = torch.arange(0, T, dtype=torch.long, device=device).unsqueeze(0)

        # 组合嵌入
        tok_emb = self.wte(input_ids)
        pos_emb = self.wpe(pos_ids)
        x = self.drop(tok_emb + pos_emb)

        if return_kvs:
            kvs = {}
        else:
            kvs = None
        # 通过所有Transformer层
        for i, block in enumerate(self.blocks):

            x = block(
                x,
                return_kvs=return_kvs,
                kv_cache=kv_cache[i] if kv_cache else None,
                mask=mask,
            )
            if return_kvs:
                kvs[i] = x['kvs']
            else:
                kvs = None
            x = x['output']
            break
        # 输出logits
        x = self.norm(x)
        logits = self.head(x)
        return {
            "logits": logits,
            "kvs": kvs
        }

    def generate(
            self,
            input_ids,
            max_new_tokens=20,
            temperature=1.0,
            use_kv_cache=True
    ):
        self.eval()
        kvs = None
        input_ids = torch.tensor(input_ids)
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # 截断输入以避免超出位置嵌入范围
                input_trunc = input_ids[:, -self.config.n_positions:]
                output = self(
                    input_ids=input_trunc,
                    return_kvs=use_kv_cache,
                    kv_cache=kvs,
                )
                kvs = output['kvs']
                logits = output['logits']
                next_token_logits = logits[:, -1, :] / temperature
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                input_ids = torch.cat([input_ids, next_token], dim=1)
        return input_ids


# 示例用法
if __name__ == "__main__":
    inputs = torch.tensor([[i for i in range(700)], [i for i in range(700)]]).to('cuda')
    config = GPT2Config()
    model = GPT2Model(config)
    print(model)
    model.to('cuda')
    start = time.time()
    output = model.generate(inputs, use_kv_cache=True, max_new_tokens=200)
    print(time.time()-start)
    output = [output[i][len(input_ids):] for i, input_ids in enumerate(inputs)]
    print(output)
