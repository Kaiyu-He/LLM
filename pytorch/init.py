import torch.nn as nn
import torch


device = 'cpu'
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


model = Model()

model.parameters()  # 获取所有可训练参数
for name, param in model.named_parameters():
    print(name, param.shape)

model.train()  # 开启训练模式（启用dropout等）
model.eval()

model.state_dict()  # 获取参数
torch.save(model.state_dict(), 'model_weights.pth')  # 保存
model.load_state_dict(torch.load('model_weights.pth'))  # 导入参数

model.zero_grad()  # 梯度清零，防止梯度累计
model.to(device)  # 模型转移到指定设备
