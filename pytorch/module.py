import torch.nn as nn
# 卷积层
nn.Conv2d(in_channels, out_channels, kernel_size， stride, padding)  # Conv1d Conv2d Conv3d

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
nn.Dropout(p=0.5)  # 丢弃率
nn.Dropout2d(p=0.5)

# embedding 层
nn.Embedding(num_embeddings, embedding_dim)  # 1000 个词表，每个词映射为 128 维向量

# 计算损失函数
nn.CrossEntropyLoss()  # 交叉熵
nn.MSELoss()  # 均方损失

layers = nn.Sequential(  # 组合多个层
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)

nn.Flatten(start_dim=1, end_dim=-1)  # 维度展平