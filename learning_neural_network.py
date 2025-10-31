
import torch
import torch.nn as nn
import torch.optim as optim


class simpleNN(nn.Module):
    def __init__(self):
        super(simpleNN, self).__init__()
        self.fc1 = nn.Linear(2,2)
        self.fc2 = nn.Linear(2,1)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = simpleNN()



# 模型网络训练
# 准备数据
X = torch.randn(10,2)
Y = torch.randn(10,1)

# 定义损失函数
criterion = nn.MSELoss()

# 定义优化器
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练过程
for epoch in range(100):
    model.train() #设置模型为训练模式
    optimizer.zero_grad()  #清除梯度
    output = model(X)
    loss = criterion(output, Y)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1} / 100, Loss {loss.item():.4f}")

# 测试与评估
model.eval()
X_test = torch.randn(10,2)
Y_test = torch.randn(10,1)
with torch.no_grad():
    output = model(X_test)
    losss = criterion(output, Y_test)
    print(f"Test Loss {losss.item():.4f}")


