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

        self.W_q = nn.Linear(d_model, n_kv_head * dim_k, bias=False)
        self.W_k = nn.Linear(d_model, n_kv_head * dim_k, bias=False)
        self.W_v = nn.Linear(d_model, n_kv_head * dim_v, bias=False)
        self.W_o = nn.Linear(n_head * dim_v, d_model, bias=False)

    def forward(
            self,
            inputs: Tensor,
            use_flash_attn: bool = False,
            mask: torch.Tensor = None
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
        # 分头
        q = self.W_q(inputs).view(B, -1, self.n_kv_head, self.dim_k).transpose(1, 2)  # (B, H, T, dim_k)
        k = self.W_k(inputs).view(B, -1, self.n_kv_head, self.dim_k).transpose(1, 2)  # (B, H, T, dim_k)
        v = self.W_v(inputs).view(B, -1, self.n_kv_head, self.dim_v).transpose(1, 2)  # (B, H, T, dim_v)

        if use_flash_attn:
            output = F.scaled_dot_product_attention(q, k, v, is_causal=False)  # (B, H, T, dim_v)
        else:
            scores = torch.matmul(q, k.transpose(-2, -1)) / (self.dim_k ** 0.5)  # (B, H, T, T)
            if mask is not None:
                scores = scores.masked_fill(mask == 0, float('-inf'))
            attn = F.softmax(scores, dim=-1)  # (B, H, T, T)
            output = torch.matmul(attn, v)  # (B, H, T, dim_v)

        output = output.transpose(1, 2).contiguous().view(B, T, self.d_model)  # (B, T, D)
        output = self.W_o(output)  # (B, T, D)
        return {
            "output": output
        }
