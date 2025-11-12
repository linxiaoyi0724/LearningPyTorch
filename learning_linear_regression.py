# 数据准备
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn

# 设置随机种子
torch.manual_seed(42)

# 生成测试数据
X = torch.randn(100,2)
true_w = torch.tensor([2.0, 3.0])
true_b = 4.0
Y = X @ true_w + true_b + torch.randn(100) * 0.1

# print(X[:5])
# print(Y[:5])

# 定义线性回归模型
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(2,1)

    def forward(self, x):
        return self.linear(x)

model = LinearRegressionModel()
# print(model)

# 定义损失函数跟优化器
critersion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)

#训练模型
num_epochs = 1000
for epoch in range(num_epochs):
    model.train() # 设置为训练模式

    # 前向传播
    prediction = model(X)
    loss = critersion(prediction.squeeze(), Y) # 压缩为1D

    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch +1) % 100 == 0:
        print(f"epoch {epoch+1}/1000, loss {loss.item():.4f}")

# 评估模型
print(f"Predicted weight: {model.linear.weight.data.numpy()}")
print(f"Predicted bias: {model.linear.bias.data.numpy()}")

# 在新数据上做预测
with torch.no_grad():
    pridections = model(X)

plt.scatter(X[:,0], Y, color='blue', label='True values')
plt.scatter(X[:,0], pridections, color='red', label='Predicted values')
plt.legend()
plt.show()