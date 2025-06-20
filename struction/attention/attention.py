import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


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

        if kv_cache is not None: # 实现 kv 缓存
            assert isinstance(kv_cache, tuple)
            k_cache, v_cache = kv_cache
            k = torch.cat((k_cache, k), dim=1)  # (B, T_ctx + 1, H, dim_k)
            v = torch.cat((v_cache, v), dim=1)  # (B, T_ctx + 1, H, dim_v)
        if return_kvs:
            kvs = (k, v)
        else:
            kvs = None

        q, k, v = map(lambda e: e.transpose(1, 2), (q, k, v))  # (B, H, T, dim_k)

        if use_flash_attn: # 注意力机制
            output = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)  # (B, H, T, dim_v)
        else:
            scores = torch.matmul(q, k.transpose(-2, -1)) / (self.dim_k ** 0.5)  # (B, H, T, T)
            if mask is not None:
                scores = scores.masked_fill(mask == 0, float('-inf'))
            attn = F.softmax(scores, dim=-1)  # (B, H, T, T)
            output = torch.matmul(attn, v)  # (B, H, T, dim_v)
        # 合并头
        output = output.transpose(1, 2).contiguous().view(B, -1, self.d_model)  # (B, T, D)
        output = self.W_o(output)  # (B, T, D)
        return {
            "output": output,
            "kvs": kvs,
        }
