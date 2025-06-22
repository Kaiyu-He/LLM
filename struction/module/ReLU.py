# 用autograd.Function实现一个激活函数是ReLU的FFN，带重计算

import torch
from torch.autograd import Function
import torch.nn as nn


class ReLU(Function):

    @staticmethod
    def forward(ctx, input_):
        # 保存用于反向传播
        ctx.save_for_backward(input_)
        # ReLU 前向传播：max(0, x)
        output = input_.clamp(min=0)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # 获取保存的输入值
        input_, = ctx.saved_tensors
        # 创建梯度掩码（输入大于0的位置为1，否则为0）
        grad_input = grad_output.clone()
        grad_input[input_ < 0] = 0
        return grad_input


class FFN(nn.Module):
    def __init__(
            self,
            d_model: int = 512,
            d_ffn: int = 128,
    ):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ffn)
        self.fc2 = nn.Linear(d_ffn, d_model)
        self.relu = ReLU.apply

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))