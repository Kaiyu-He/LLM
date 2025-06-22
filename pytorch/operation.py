import torch

# 张量创建

m = torch.tensor([[1, 2], [3, 4]]) # 直接初始化
rand_tensor = torch.rand(2, 3, 4)  # 3维张量（形状：2×3×4） 随机张量（均匀分布）
zeros = torch.zeros(2, 3)  # 全零张量
ones = torch.ones(2, 3)    # 全一张量
eye = torch.eye(3)         # 单位矩阵
B = torch.tensor((1, 2, 3), dtype=torch.float32, requires_grad=True)

# 转下三角矩阵
torch.tril(torch.ones(4,4))


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

# >>> tensor([0, 2, 4, 6, 8])

L = torch.linspace(start=1, end=10, steps=5)
print(L)

# >>> tensor([ 1.0000,  3.2500,  5.5000,  7.7500, 10.0000])

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

# 复制张量为二维
expanded_row = x.unsqueeze(0).expand(4, -1)  # 复制为4行，列数不变

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

# >>> tensor([[1., 2.], [3., 4.]])
# >>> tensor([4., 6.])
# >>> tensor(10.)

A.max() # 求最大值
A.argmax()
torch.sort(A) # 元素排序
torch.topk(A, 3)
torch.mean(A, dim=1, keepdim=True)

# keepdim=True：对应行输出

# keepdim=False：转变为一维的tensor输出。

torch.std(A) # 求标准差