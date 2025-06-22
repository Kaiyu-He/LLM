import torch
from torch import Tensor

def make_causal_mask(inputs: Tensor):
    batch, seq_len = inputs.size()
    mask = torch.tril(torch.ones(seq_len, seq_len))
    return mask

# 如何使用
# mask = make_causal_mask(inputs)
# score = score.masked_fill(mask == 0, float('-inf'))
