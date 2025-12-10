import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader,TensorDataset
import numpy as np

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        # 定义RNN层
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        # 定义全连接层
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self,x):
        out,_ = self.rnn(x)
        out = out[:,-1,:]
        out = self.fc(out)
        return out

# 生成一些随机序列数据
num_samples = 1000
seq_len = 10
input_size = 5
output_size = 2

# 随机生成一些输入数据
X = torch.randn(num_samples,seq_len, input_size)
# 随机生成一些目标标签
Y = torch.randint(0,output_size, (num_samples,))

# 创建数据加载器
dataset = TensorDataset(X,Y)
train_loader = DataLoader(dataset=dataset,batch_size=32,shuffle=True)


model = SimpleRNN(input_size=input_size,hidden_size=64,output_size=output_size)
# 定义损失函数跟优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=0.001)

# 训练模型
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        outputs = model(inputs)
        loss = criterion(outputs,labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # 计算准确率
        _, predicted = torch.max(outputs,1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    accuracy = 100.0 * correct / total
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss / len(train_loader):.4f}, Accuracy: {accuracy:.2f}%")


# 测试模型
model.eval()
with torch.no_grad():
    total = 0
    correct = 0
    for inputs, labels in train_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs,1)
        total += labels.size(0)
        correct += (predicted==labels).sum().item()
    accuracy = 100.0 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
