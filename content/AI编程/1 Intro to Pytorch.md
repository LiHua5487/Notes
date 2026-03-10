
用 pytorch 实现一个简单的 LeNet

![[AI编程/imgs/img1/image.png]]

pytorch 中的 Tensor 指的是一个多维的 array 

```python
import torch 
import numpy as np 

data = [[1, 2], [3, 4]] 
x_data = torch.tensor(data) # 用列表初始化tensor

np_array = np.array(data) # 用列表初始化array
x_np = torch.from_numpy(np_array) # 用array初始化tensor
x_rand = torch.rand_like(x_data, dtype=torch.float) # 随机初始化
x_ones = torch.ones_like(x_data) # 所有元素都是1，形状与x_data相同

tensor = tensor.to('cuda') # 将数据传到GPU上
```

首先需要加载数据集，以 CIFAR10 为例

```python
import torch
import torchvision

batch_size = 4
# 定义初始化变换，先把数据转为tensor，且要归一化，均值方差设为0.5
transform = torchvision.transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
# 获取数据集
trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform)
# 加载到DataLoader
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=batch_size, shuffle=True, num_workers=2)

# 获取一个数据
dataiter = iter(trainloader) # 设为迭代器
images, labels = next(dataiter) # 进行迭代
```

>pytorch 的 DataLoader 支持
>1. 自动批处理：自动从数据集中取出的单个样本组合成一个 batch
>2. 数据打乱：在每个训练周期 epoch 开始时打乱数据的顺序
>3. 多线程并行与内存优化：支持CPU加载与GPU计算异步

而后要实现模型

```python
class LeNet(torch.nn.Module):
    def __init__(self): # 定义各层
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x): # 前向传播
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten dimensions
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
```

pytorch 中的自动微分如下

```python
net = LeNet() # 创建模型实例
print(net)

input = torch.randn(1, 1, 32, 32) # (batchsize,channel,H,W)
out = net(input) # pytorch对模型进行封装，自动进行前向传播

params = list(net.parameters()) # 模型参数

# 损失函数
criterion = nn.CrossEntropyLoss() 
loss = criterion(out, target) 

net.zero_grad() # 梯度清零
loss.backward(torch.randn(1, 10)) # 反向传播
```

接着实现反向传播

```python
learning_rate, weight_decay = 0.001, 0.0005
for weights in net.parameters():
    grad = weights.grad + 2 * weights * weight_decay
    weights.data = weights.data - grad * learning_rate
```

模型参数是 tensor 及其梯度，下面是一些基本运算

```python
x = torch.rand(5, 5)
y = torch.rand((5, 5), requires_grad=True) # 可以储存关于y的梯度
a = x + y
out = a.sum()
out.backward()
print(y.grad)
```

打印结果如下

```powershell
>>> y.grad
tensor([[1., 1., 1., 1., 1.],
        [1., 1., 1., 1., 1.],
        [1., 1., 1., 1., 1.],
        [1., 1., 1., 1., 1.],
        [1., 1., 1., 1., 1.]])
```

可以在 cuda 中实现一些基本的运算，用 python 封装







