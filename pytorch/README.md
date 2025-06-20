# python

## 张量操作

```python
import torch

# 张量创建

m = torch.tensor([[1, 2], [3, 4]]) # 直接初始化
rand_tensor = torch.rand(2, 3, 4)  # 3维张量（形状：2×3×4） 随机张量（均匀分布）
zeros = torch.zeros(2, 3)  # 全零张量
ones = torch.ones(2, 3)    # 全一张量
eye = torch.eye(3)         # 单位矩阵
B = torch.tensor((1, 2, 3), dtype=torch.float32, requires_grad=True)

# 构建相同类型的张量

print(torch.zeros_like(m))  # 零张量
print(torch.ones_like(m))  # 单位张量
print(torch.rand_like(m))  # 随机张量

# 构建正太分布

torch.manual_seed(111)

# 均值分别为1、2、3、4，标准差分别为1、2、3、4

print(torch.normal(mean=torch.arange(1, 5.0), std=torch.arange(1, 5.0)))

K = torch.arange(start=0, end=10, step=2)
print(K)

>>> tensor([0, 2, 4, 6, 8])

L = torch.linspace(start=1, end=10, steps=5)
print(L)

>>> tensor([ 1.0000,  3.2500,  5.5000,  7.7500, 10.0000])

# 张量查询

print(x.shape)   # 形状：torch.Size([2, 2])
print(x.dtype)   # 数据类型：torch.int64
print(x.numel())  # 查看元素数量
a.long() # 类型转换

# 计算梯度

B = torch.tensor((1, 2, 3), dtype=torch.float32, requires_grad=True)
Y = B.pow(2).sum()
Y.backward() # 反向传播
print(B.grad) # 输出指定参数的梯度

# 与 numpy 切换

x = torch.from_numpy(x) # 转tensor
x = x.numpy() # 转 numpy

# 改变形状 .reshape

A = torch.arange(12.0).reshape(3,4)

# 插入维度

A = torch.unsqueeze(A, dim=0)

# 移除维度

A = torch.squeeze(A, dim=0)

# 维度扩充

A = A.expand(3, -1)

A = torch.tensor([1, 2, 3])
A.repeat(1, 2, 2) # 三个参数分别代表三个维度repeat的次数

# 张量拼接

torch.cat((A, B), dim=0) # 给定维度中张量拼接
torch.stack((A, B), dim=1) # 沿新维度连接张量————形状必须一致 dim为的大小是2

# 张量分块

B1, B2 = torch.chunk(A, 2, dim=0) # 若不能整除时，则最后一块将最小
D1, D2, D3 = torch.split(A, [1, 2, 3], dim=1) # 将张量分块，可指定每一块的大小（示例中大小为 1，2，3）

# 张量运算

# 比较

torch.eq(A,B) # 判断相等———每个元素 tensor([True, True, True, True, True, True])
torch.equal(A,B) # 判断相等———整体 True

torch.ge(A, B) # 比较大于等于
torch.gt(A, B) # 比较大于
torch.isnan() # 判断是否为缺失值

# 四则运算 （对应元素计算）

print(A + B)
print(A - B)
print(A * B) # 点成
print(A / B)

torch.pow(A, 3) # 求幂
A ** 3
torch.sqrt(A) # 开根号
A ** 0.5

torch.exp(A) # 指数
torch.log(A) # 对数
torch.t(A) # 转置
torch.matmul(A, B) # 矩阵乘法
torch.inverse(A) # 矩阵求逆

# 元素求和

print(x)
print(sum(x))
print(torch.sum(x))

> > > tensor([[1., 2.],
> > > [3., 4.]])
> > > tensor([4., 6.])
> > > tensor(10.)

A.max() # 求最大值
A.argmax()
torch.sort(A) # 元素排序
torch.topk(A, 3)
torch.mean(A, dim=1, keepdim=True)

# keepdim=True：对应行输出

# keepdim=False：转变为一维的tensor输出。

torch.std(A) # 求标准差
```

## torch.nn.module 类 网络模块操作

### 常见模块

```python
# 卷积层
nn.Conv2d(in_channels, out_channels, kernel_size， stride, padding) # Conv1d Conv2d Conv3d
 
# 线性层
nn.Linear(in_features, out_features, bias=False)

# 池化层
nn.MaxPool1d/2d/3d(kernel_size)
nn.AvgPool1d/2d/3d(kernel_size)

# 归一化层
nn.LayerNorm(normalized_shape)
nn.BatchNorm1d/2d/3d(num_features)

# 激活函数
nn.ReLU()
nn.Sigmoid()
nn.Tanh()
nn.Softmax(dim=2)(a)

# dropout 层 
nn.Dropout(p=0.5) # 丢弃率
nn.Dropout2d(p=0.5)

# embedding 层
nn.Embedding(num_embeddings, embedding_dim)  # 1000 个词表，每个词映射为 128 维向量

# 计算损失函数
nn.CrossEntropyLoss() # 交叉熵
nn.MSELoss() # 均方损失

layers = nn.Sequential( # 组合多个层
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)

nn.Flatten(start_dim=1, end_dim=-1) # 维度展平
```

### 定义模块

```python
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        # 定义网络层
        self.layer1 = nn.Linear(in_features=784, out_features=256)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(in_features=256, out_features=10)

    def forward(self, x):
        # 定义前向传播（模型网络结构）
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x
```

### 常用方法和属性

```python
model = Model()

model.parameters() # 获取所有可训练参数
for name, param in model.named_parameters():
    print(name, param.shape)
  
model.train() # 开启训练模式（启用dropout等）
model.eval()

model.state_dict() # 获取参数
torch.save(model.state_dict(), 'model_weights.pth') # 保存
model.load_state_dict(torch.load('model_weights.pth')) # 导入参数

model.zero_grad() # 梯度清零，防止梯度累计
model.to(device) # 模型转移到指定设备
```

## 模型推理

```python
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
```

## 模型训练框架

## 常用模型结构

### 多头注意力机制

$O(n^2 d)$

```python
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
        k = self.W_k(inputs).view(B, -1, self.n_kv_head, self.dim_k).transpose(1, 2) # (B, H, T, dim_k)
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
```

$O(nd)$

### KV Cache具体实现

```python
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
```

```python
class FFN(nn.Module):
    def __init__(
            self,
            embed_dim: int = 128,
            hidden_dim: int = 128,
    ):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.gelu = nn.GELU()

    def forward(self, x):
        return self.fc2(self.gelu(self.fc1(x)))
```
