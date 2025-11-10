'''
# 创建自己的数据集
import torch
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, X_data, Y_data):
        self.X_data = X_data
        self.Y_data = Y_data

    def __len__(self):
        return len(self.X_data)

    def __getitem__(self, idx):
        x = torch.tensor(self.X_data[idx], dtype=torch.float)
        y = torch.tensor(self.Y_data[idx], dtype=torch.float)
        return x, y

X_data = [[1,2], [3,4], [5,6], [7,8]]
Y_data = [1,0,1,1]
my_dataset = MyDataset(X_data, Y_data)


# 使用DataLoader加载数据
from torch.utils.data import DataLoader
dataloader = DataLoader(my_dataset, batch_size=2, shuffle=True)
for epoch in range(4):
    for batch_idx, (inputs, lables) in enumerate(dataloader):
        print(f"Batch {batch_idx +1 }:")
        print(f"Inputs: {inputs}")
        print(f"Labels: {lables}")

'''
from torch.utils.data import DataLoader

'''
# 预处理与数据增强
import torchvision.transforms as transforms
from PIL import Image

# 定义数据处理流水线
transforms = transforms.Compose([
    transforms.Resize((128, 128)),  # 调整图片大小
    transforms.ToTensor(),          # 将图片调整为张量
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) #标准化
])

# 加载图片
image = Image.open("./test_data/1.jpeg")

image_tessor = transforms(image)
print(image_tessor.shape)
'''


'''
from torchvision.transforms import transforms
# 图像数据增强
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),  # 水平随机翻转
    transforms.RandomRotation(30),      # 随机旋转30度
    transforms.RandomResizedCrop(128), transforms.ToTensor(), # 随机裁剪并调整为128 * 128
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

'''



'''
# 加载图像数据集
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# 定义预处理操作
transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,),(0.5,))
])

# 下载并加载MNIST 数据集
train_dataset = datasets.MNIST(root='./test_data', train=True, transform=transforms, download=True)
test_dataset = datasets.MNIST(root='./test_data', train=False, transform=transforms, download=True)

# 创建 DataLoader
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 迭代训练数据
for inputs, labels in train_loader:
    print(inputs.shape)
    print(labels.shape)

'''



# 使用多个数据源
from torch.utils.data import ConcatDataset
combined_dataset = ConcatDataset([data_set_1, data_set_2])
combined_loader = DataLoader(combined_dataset, batch_size=64, shuffle=True)