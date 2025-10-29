
'''
import torch
import torch.nn as nn
import torch.optim as optim

# 定义一个简单的神经网络
class simpleNN(nn.Module):
    def __init__(self):
        super(simpleNN, self).__init__()
        self.fc1 = nn.Linear(2, 2)
        self.fc2 = nn.Linear(2, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 创建模型实例
model = simpleNN()

# 随机输入
x = torch.randn(1, 2)

# 前向传播
y = model(x)
print(y)

# 定义损失函数
criterion = nn.MSELoss()

# 假设目标值为1
target = torch.randn(1, 1)
print(target)
loss = criterion(y, target)
print(loss)

# 定义优化器
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练步骤
optimizer.zero_grad() #清空梯度
loss.backward()  #反向传播
optimizer.step() #更新参数
'''



import torch
import torch.nn as nn
import torch.optim as optim

# 定义模型类及其网络结构
class simpleNN(nn.Module):
    def __init__(self):
        super(simpleNN, self).__init__()
        self.fc1 = nn.Linear(2,2)
        self.fc2 = nn.Linear(2,1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 创建模型实例
model = simpleNN()
model.to(device)

# 定义损失函数及优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 设置数据跟标签
x = torch.randn(10, 2)
y = torch.randn(10, 1)

x.to(device)
y.to(device)

# 训练循环
for epoch in range(100):
    optimizer.zero_grad()
    ouput = model(x)
    loss = criterion(ouput, y)
    loss.backward()
    optimizer.step()

    #每10轮输出一次损失
    if(epoch + 1) % 10 == 0:
        print(f'Epoch {epoch + 1} | Loss {loss.item():.4f}')



