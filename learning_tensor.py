import torch
import numpy as np
'''
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
'''


'''
# 张量的操作
tensor = torch.tensor([[1,2,3], [4,5,6]], dtype=torch.float32)
print(f"原始tensor: {tensor}")

# 索引跟切片操作
print(f"获取第一行: {tensor[0]}")
print(f"获取第一行第一列的元素: {tensor[0][0]}")
print(f"获取第二列所有的元素: {tensor[:,1]}")

# 形状变换操作
reshaped = tensor.view(3,2)
print(f"改变形状后的张量: {reshaped}")
flattened = tensor.flatten()
print(f"展平后的张量: {flattened}")

# 数学运算操作
tensor_add = tensor + 10
print(f"张量+10:{tensor_add}")
tensor_mul = tensor * 2
print(f"张量*2: {tensor_mul}")
tensor_sum = tensor.sum()
print(f"张量的元素和: {tensor_sum}")

# 与其他张量的操作
tensor2 = torch.tensor([[5,6,7],[7,8,9]],dtype=torch.float32)
print(f"另一个张量: {tensor2}")
tensor_dot = torch.matmul(tensor, tensor2.T)
print(f"张量相程的结果为: {tensor_dot}")

# 条件判断和筛选
mask = tensor > 3
print(f"大于3的布尔掩码: {mask}")
filter_tensor = tensor[mask]
print(f"大于3的元素: {filter_tensor}")
'''



# 张量的GPU加速
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
x = torch.tensor([1,2,3,4,5,6,7,8,9], dtype=torch.float32, device=device)
print(torch.cuda.is_available())


# 张量与numpy的互操作
numpy_array = np.array([[1,2,3],[4,5,6]])
print(f"numpy_arary: {numpy_array}")

tensor_numpy_arary = torch.from_numpy(numpy_array)
print(f"tensor_numpy_arary: {tensor_numpy_arary}")

# 内存共享
numpy_array[0,0] =10
print(f"numpy_arary: {numpy_array}")
print(f"tensor_numpy_arary: {tensor_numpy_arary}")

tensor = torch.tensor([[5,6,7],[7,8,9]])
print(f"tensor: {tensor}")
numpy_tensor = tensor.numpy()
print(f"numpy_tensor: {numpy_tensor}")

tensor[0,0] = 10
print(f"tensor: {tensor}")
print(f"numpy_tensor: {numpy_tensor}")

# 不共享内存的情况
tensor_independent = torch.tensor([[1,3,5],[2,4,8]],dtype=torch.float32)
numpy_independent = tensor_independent.clone().numpy()
print(f"numpy_independent: {numpy_independent}")
tensor_independent[0,0] = 10
print(f"numpy_independent: {numpy_independent}")
