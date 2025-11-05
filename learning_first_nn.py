'''
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
'''



'''
# 可视化代码
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

#定义输入层大小，隐藏层大小,输出层大小与批次大小
n_i, n_h, n_o, batch_size = 10, 5, 1, 10

# 创建虚拟数据
X = torch.randn(batch_size, n_i);
Y = torch.tensor([[1.0], [0.0], [0.0],
                  [1.0], [1.0], [1.0], [0.0], [0.0], [1.0], [1.0]])

# 构建网络结构
model = nn.Sequential(
    nn.Linear(n_i, n_h),
    nn.ReLU(),
    nn.Linear(n_h, n_o),
    nn.Sigmoid()
)

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr = 0.001)

losses = []

for epoch in range(100):
    y_pred = model(X)
    loss = criterion(y_pred, Y)
    losses.append(loss.item()) #记录损失值
    print(f"Epoch {epoch}/100: Loss {loss.item():.4f}")
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# 可视化损失函数变化曲线
plt.figure(figsize = (8,5))
plt.plot(range(1,101), losses, label='Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss over epochs')
plt.legend()
plt.grid()
plt.show()

y_pred_final = model(X).detach().numpy()
y_actual = Y.numpy()

plt.figure(figsize = (8,5))
plt.plot(range(1, batch_size + 1), y_actual, 'o-', label='Actual', color='blue')
plt.plot(range(1, batch_size + 1), y_pred_final, '-', label='Predicted', color='red')
plt.xlabel('sample index')
plt.ylabel('value')
plt.title('Actual vs predicted values')
plt.legend()
plt.grid()
plt.show()
'''











import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

n_samples = 100
data = torch.randn(n_samples, 2)
labels = (data[:,0]**2 + data[:,1]**2 < 1).float().unsqueeze(1)

plt.scatter(data[:,0], data[:,1], c=labels.squeeze(), cmap='coolwarm')
plt.title('Generated Data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()


class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(2,4)
        self.fc2 = nn.Linear(4,1)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        x = torch.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

model = SimpleNet()

criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr = 0.1)

epochs = 100
for epoch in range(epochs):
    output = model(data)
    loss = criterion(output, labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch {epoch+1}/100: Loss {loss.item():.4f}')




def plot_decision_boundary(model, data):
    x_min, x_max = data[:, 0].min() - 1, data[:,0].max() + 1
    y_min, y_max = data[:, 1].min() - 1, data[:,1].max() + 1
    # 生成网格点
    xx, yy = torch.meshgrid(torch.arange(x_min, x_max, 0.1), torch.arange(y_min, y_max, 0.1), indexing='ij')
    grid = torch.cat([xx.reshape(-1, 1), yy.reshape(-1,1)], dim=1)
    prediction = model(grid).detach().numpy().reshape(xx.shape)
    plt.contourf(xx, yy, prediction, levels=[0, 0.5, 1], cmap='coolwarm', alpha=0.7)
    plt.scatter(data[:,0], data[:,1], c=labels.squeeze(), cmap='coolwarm',edgecolors='k')
    plt.title("Decision Boundary")
    plt.show()

plot_decision_boundary(model, data)










