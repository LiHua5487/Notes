
# 项目结构

项目中主要包含以下文件
- `./autodiff` 文件夹
	- `autograd.py` ：自动微分相关代码
	- `layer.py` ：神经网络层的定义与封装
	- `optimizer.py` ：优化器类
	- `module.py` ：定义了模块类 `Module` ，是 `Sequential` 类、各种神经网络层类、自定义的神经网络类的基类
- `./mytensor` 文件夹
	- `MyTensor.cu` ：自己实现的 Tensor 类与基本操作
	- `MyGo.cu` ：神经网络运算的核函数与封装函数
	- `setup_mytensor.py / setup_mygo.py` ：编译上面的 cuda 文件为 python 库
- `train.py` ：训练程序，定义了模型结构与训练过程等
- `train_pytorch.py` ：用 pytorch 实现的训练程序，用作对比

# 运行方式

`MyGo.cu` 中使用了 `cublas` ，由于我的电脑是 CUDA 13.0 的版本，尽管 conda 环境已经安装低版本的 `cudatoolkit` ，但编译后依赖项仍是 CUDA 13.0 版本的 dll ，所以我在 `import` 前指定了 dll 路径（绝对路径），这导致不同的环境可能无法正常 `import` 

运行 `train.py` 以训练或测试模型，参数如下（需在文件内修改）
- `mode` ：`'TRAIN'` 只训练，`'TEST'` 只测试，`'ALL'` 先训练再测试
- `save_path` ：模型保存与加载的路径

使用 `python file_name.py build` 指令运行 `setup_mytensor.py / setup_mygo.py` 以编译 cuda 文件，编译后需要把 `.pyd` 文件放到根目录中

# 结果分析

我定义了简单的卷积网络 `ConvNet` 类，参数配置如下

```
BATCH_SIZE = 128
EPOCHS = 20
lr = 0.001
save_path = "./model.pkl"
model = ConvNet()
optimizer = AdamWOptimizer(model, lr=lr)
criterion = layer.CrossEntropyLoss()
```

训练结果如下，其中首轮 epoch 时间 37s ，后续 epoch 均在 18s~20s

```
Training completed! Best test accuracy: 77.84%
Loading best model for final evaluation...
Model loaded from ./model.pkl
Final test accuracy with best model: 77.84%
```

下面是 pytorch 在相同模型与参数下的训练结果，每个 epoch 在 11s~12s

```
PyTorch Training completed!
Total training time: 284.97 seconds
Average time per epoch: 14.25 seconds
Best test accuracy: 74.86%

Loading best model for final evaluation...
Final test accuracy with best model: 74.86%
```

---

我还定义了一个简单的 ResNet 网络，但是相比之下训练速度很慢，除首轮 epoch 外，每个 epoch 在 46s~54s ，训练 50 个 epoch 后的结果如下

```
Loading best model for final evaluation...
Model loaded from ./model_resnet.pkl
Final test accuracy with best model: 77.39%
```

# 实现细节

## MyTensor 的补充

在先前 lab 的基础上，我加入了逐元素运算、标量和向量间的运算、reshape 等函数，以便在反向传播时使用，这样就不需要转回 CPU 计算

## MyGo 的调整

修改卷积函数接口，支持 `kernel_size` `stride` `padding` 参数，因为 ResNet 需要用到 1×1 卷积

加入了 BatchNorm1d 和 2d 函数，实现 FC 层和卷积层的 BatchNorm 操作

>为了实现 BN ，我实现了 `.train()` 和 `.eval()` 切换训练和预测模式，因为 BN 层在两种模式下表现不同

加入了全局平均池化函数，用于实现 ResNet 

## 自动微分

我参考 lab5 的框架，实现了简单的自动微分框架，能自动构建计算图并进行反向传播，相关代码在 `./autodiff` 目录下，主要涉及两个文件
- `autograd.py` ：定义了基类 `Op` 和 `Value` ，`Value` 的子类 `TensorNode` （计算图节点），相关的前向与反向运算等
- `layer.py` ：定义了一系列神经网络层，进行了封装，便于调用

## 优化器

我实现了 SGD + momentum 和 AdamW 优化器类，相关代码在 `./autodiff/optimizer.py` 中
- 提供 `step` 函数用于更新模型参数
- 主要计算均使用 `MyTensor` 中的 `Tensor` 类，在 GPU 上完成
- 没有 `zero_grad` 函数，因为这个函数在 `Module` 类中实现了

## Module 封装

在 `./autodiff/module.py` 中定义了 `Module` 基类，是 `Sequential` 类、各种神经网络层类、自定义的神经网络类的基类，支持以下功能
- 切换训练与评估模式
- 自动获取参数（全部参数，用于保存与加载模型；可训练参数，用于优化器更新）
	- 包含 `_modules` 成员，是一个字典，将模块名称映射到类实例
	- 重载了 `__setattr__` 函数，在定义 `Module` 类型（包括其派生类）的成员时，会自动将其加入 `_modules` 中
	- 遍历 `_modules` 获取各模块的参数
- 支持清空梯度
- 提供 `forward` 虚函数

`Sequential` 类的功能类似于 `nn.Sequential` 
- 支持便捷的定义模型结构，使用 `Sequential(module1, module2, module3, ...)` 的格式初始化
	- 初始化时，会将初始化参数 `(module1, module2, module3, ...)` 加入 `_modules` 中
	- 自身不含其它模块成员，其包含的模块仅在初始化时确定
- 自动获取参数（继承自基类）
- 自动进行前向传播（依次遍历每个层）

而各种神经网络层中重载了获取参数的方法，直接返回自身的参数，而非遍历 `_modules` 

## 其它细节

在参数初始化时，我使用了 kaiming uniform 方法进行初始化，以提高训练效果与稳定性

使用了数据增强

对于数据加载器，我调整了参数配置，使用了下面几个带有注释的参数，这样调整后，仅有首个 epoch 时间较长（因为需要启动加载器），后续 epoch 的时间明显缩短

```
trainDataLoader = torch.utils.data.DataLoader(
        dataset=trainData, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=4, # 使用4个工作进程加载数据
        persistent_workers=True, # 保持工作进程活跃，避免重复创建
        pin_memory=True # 将数据固定在内存中，加速GPU传输
    )
```












