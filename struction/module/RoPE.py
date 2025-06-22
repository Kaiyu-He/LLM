# hekaiyu
# 2025-6-17

from typing import Optional, Tuple
import torch


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """
    预计算旋转位置编码所需的频率矩阵，使用复数指数形式表示
    参数:
        dim: 模型隐藏层维度（通常为注意力头的维度）
        end: 预计算的最大位置索引（序列长度）
        theta: 基础频率缩放因子，控制不同维度的频率分布
    """
    # 计算每个偶数维度的基础频率: 1/θ^(2i/d)
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))

    # 创建位置索引张量 [0,1,2,...,end-1]，表示序列中的每个位置
    t = torch.arange(end, device=freqs.device)

    # 外积计算每个位置和维度组合的频率值，形成矩阵 [end, dim//2]
    # 每行对应一个位置，每列对应一个维度的频率
    freqs = torch.outer(t, freqs).float()

    # 计算每个频率值的余弦和正弦值，用于后续旋转操作
    cos, sin = freqs.cos(), freqs.sin()

    # 堆叠余弦和正弦值形成2x2旋转矩阵: [[cos, -sin], [sin, cos]]
    # 最终形状为 [end, dim//2, 2, 2]，其中每个2x2子矩阵表示一个旋转操作
    return torch.stack((cos, -sin, sin, cos), dim=-1).view(*freqs.size(), 2, 2)


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor, seq_dim: int):
    """
    重塑频率矩阵，使其能够与输入张量进行广播乘法

    参数:
        freqs_cis: 预计算的频率矩阵 [end, dim//2, 2, 2]
        x: 输入张量（查询Q或键K）
        seq_dim: 序列维度在输入张量中的索引（通常为1）
    """
    # 获取输入张量的维度数
    ndim = x.ndim

    # 确保序列维度索引有效
    assert 0 <= seq_dim < ndim

    # 确保频率矩阵形状与输入张量兼容
    assert freqs_cis.shape == (
        x.shape[seq_dim],  # 序列长度必须匹配
        x.shape[-3],  # 倒数第三维（通常为注意力头数）必须匹配
        2, 2  # 旋转矩阵维度
    ), f"频率矩阵形状 {freqs_cis.shape} 与输入张量形状 {x.shape} 不兼容"

    # 构建重塑形状：在非序列维度和倒数第三维度保留原始大小，其他维度设为1
    # 用于PyTorch的广播机制，避免显式循环
    shape = [
                d if i == seq_dim or i == ndim - 3 else 1
                for i, d in enumerate(x.shape[:-2])
            ] + [2, 2]  # 保留旋转矩阵的最后两个维度

    # 重塑频率矩阵以适配广播乘法
    return freqs_cis.view(*shape)


def apply_rotary_emb(
        xq: torch.Tensor,  # 查询张量 [batch, seq_len, ..., dim]
        xk: torch.Tensor,  # 键张量 [batch, seq_len, ..., dim]
        seq_dim: int,  # 序列维度索引
        freqs_cis: torch.Tensor,  # 预计算的频率矩阵
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    应用旋转位置编码到查询(Q)和键(K)张量

    参数:
        xq: 查询张量
        xk: 键张量
        seq_dim: 序列维度索引
        freqs_cis: 预计算的频率矩阵
    """
    # 打印输入张量形状用于调试
    print("查询张量形状:", xq.shape)
    print("键张量形状:", xk.shape)

    # 重塑查询张量为复数形式 [batch, seq_len, ..., dim/2, 1, 2]
    # 最后一维的2表示复数的实部和虚部
    xq_ = xq.reshape(*xq.shape[:-1], -1, 1, 2)

    # 同理重塑键张量
    xk_ = xk.reshape(*xk.shape[:-1], -1, 1, 2)

    # 重塑频率矩阵以适配查询张量的形状，便于广播乘法
    # 形状变为 [1, seq_len, 1, dim/2, 2, 2]
    freqs_cis = reshape_for_broadcast(
        freqs_cis, xq_, seq_dim
    ).float()

    # 执行复数乘法：[batch, seq_len, ..., dim/2, 1, 2] × [1, seq_len, 1, dim/2, 2, 2]
    # 乘法后对最后一维求和（合并实部和虚部计算结果），并展平倒数第三和第四维度
    xq_out = (xq_ * freqs_cis).sum(5).flatten(3)
    xk_out = (xk_ * freqs_cis).sum(5).flatten(3)

    # 恢复原始数据类型
    return xq_out.type_as(xq), xk_out.type_as(xk)


class RoPE(torch.nn.Module):
    """
    旋转位置编码模块，用于在Transformer中应用RoPE
    """

    def __init__(self, theta: float, head_dim: int, max_seqlen: int = 1024):
        """
        初始化RoPE模块

        参数:
            theta: 基础频率缩放因子
            head_dim: 注意力头的维度
            max_seqlen: 支持的最大序列长度
        """
        super().__init__()
        self.theta = theta
        self.head_dim = head_dim
        self.max_seqlen = max_seqlen

        # 预计算频率矩阵并注册为非持久缓冲区（不参与模型参数保存）
        # 形状为 [max_seqlen, head_dim//2, 2, 2]
        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(dim=head_dim, end=max_seqlen, theta=theta),
            persistent=False,
        )

    def forward(
            self, seqlen: Optional[int] = None, tok_idx: Optional[torch.Tensor] = None
    ):
        """
        返回对应位置的旋转矩阵

        参数:
            seqlen: 连续序列长度（整数）
            tok_idx: 每个token的位置索引（一维整数张量，用于非连续位置）
        """
        # 确保至少提供seqlen或tok_idx中的一个
        assert (seqlen is not None) or (tok_idx is not None), "必须提供seqlen或tok_idx"

        if tok_idx is not None:
            # 如果提供了位置索引，返回对应位置的频率矩阵
            return self.freqs_cis[tok_idx]
        elif seqlen is not None:
            # 如果提供了序列长度，返回前seqlen个位置的频率矩阵
            return self.freqs_cis[0:seqlen]