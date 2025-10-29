import torch
import numpy as np

# 从python 列表或者numpy 创建tensor
x = torch.tensor([[1,2], [3,4]])
print(x)

# 创建一个序列tensor
x = torch.arange(0,12,2)
print(x)

x = torch.linspace(0,1,5)
print(x)

# 将numpy数组转换成tensor
x = torch.from_numpy(np.array([1,2,3]))
print(x)

# 将一个数组转成tensor
x = torch.tensor([1,2,3])
print(x)

# 将numpy 数组转换tensor
x = torch.from_numpy(np.array([1,2,3]))
print(x)

# 创建2D张量
tensor_2d = torch.tensor([[1,2,3],[3,4,5],[4,5,6]])
print(tensor_2d)
print(f"tensor_2d shape is{tensor_2d.shape}")

# 创建3D张量
tensor_3d = torch.stack([tensor_2d, tensor_2d+5, tensor_2d-10])
print(tensor_3d)
print(f"tensor_3d shape is{tensor_3d.shape}")

# 创建4D张量
tensor_4d = torch.stack([tensor_3d, tensor_3d+100])
print(tensor_4d)
print(f"tensor_4d shape is{tensor_4d.shape}")


# 创建一个张量
tensor_data = torch.tensor([[1,2,3],[4,5,6]], dtype=torch.float32)
print(tensor_data)
print(f" tensor_data shape is {tensor_data.shape}")
print(f" tensor_data size is {tensor_data.size()}")
print(f" tensor_data type is {tensor_data.dtype}")
print(f" tensor_data device is {tensor_data.device}")
print(f" tensor_data dimension is {tensor_data.dim()}")
print(f" tensor_data numbers is {tensor_data.numel()}")
print(f" requires grad: {tensor_data.requires_grad}")
print(f" tensor_data is cuda:{tensor_data.is_cuda}")
print(f" tensor_data is contiguous: {tensor_data.is_contiguous()}")

# 获取单元素值
single_data = torch.tensor(32);
print(single_data.item())

tensor_data_T = tensor_data.T
print(tensor_data_T)
