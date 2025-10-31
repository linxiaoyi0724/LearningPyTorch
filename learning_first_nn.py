import torch
import torch.nn as nn
import torch.optim as optim

n_i, n_h, n_o, batch_size = 10, 5, 1, 10

x = torch.randn(batch_size, n_i)
y = torch.tensor([[1.0], [0.0], [0.0], [1.0], [1.0], [1.0], [0.0], [0.0], [1.0], [1.0]]) # 目标数据

# 创建网络
model = nn.Sequential(
    nn.Linear(n_i, n_h),
    nn.ReLU(),
    nn.Linear(n_h, n_o),
    nn.Sigmoid()
)

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(),lr=0.001)

for epoch in range(1000):
    output = model(x) #前向传播
    loss = criterion(output, y) #计算loss
    if epoch % 10 == 0:
        print(f"Epoch {epoch}/100: Loss {loss.item():.4f}")

    optimizer.zero_grad() #梯度清0
    loss.backward() #反向传播
    optimizer.step() #更新模型参数