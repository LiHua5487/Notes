---
date: 2025-03-26
tags:
  - CV
---

# CNN Training

采用 mini-batch SGD 训练 CNN 时，不断重复以下过程

1. 随机打乱训练数据 shuffle，通过数据加载器 dataloader 取出一个 batch
2. 前向传播，得到预测并计算损失
3. 反向传播，计算各个参数的梯度
4. 由优化器根据梯度更新参数

数据集通常会在每轮遍历前重新打乱，减少样本排列顺序对训练的影响

## Data Preparation

![[CV导论/imgs/img5/image.png]]

这里 $X$ 是 $N\times D$ 的数据矩阵，每行是一个样本，$N$ 是 batch size，$D$ 是每个样本的特征维数；对于图像，还可以额外保留高、宽和通道这些维度

如果输入整体偏正、偏负，或者不同特征的数值尺度相差很大，可能让后面的激活和优化更难处理，因此常在训练前做中心化或标准化

![[CV导论/imgs/img5/image-2.png]]

有时候，如果不归一化，分类结果会对权重变化很敏感

![[CV导论/imgs/img5/image-1.png|700x353]]

**中心化 centering**是减去均值，**标准化 standardization**还会除以标准差，均值表示数值大致集中在哪，标准差表示围绕均值的波动幅度，方差就是标准差的平方

对于 RGB 图像，一种常见做法是分别统计训练集中每个 channel 在所有图像、所有空间位置上的均值和标准差，再对各通道做同样的变换

$$
x'_{c}=\frac{x_c-\mu_c}{\sigma_c+\epsilon}
$$

$\epsilon$ 是避免除以 0 的小量，验证集和测试集使用训练集得到的统计量，不能各自重新统计整套测试数据

有的模型只减去各通道均值，有的还会除以标准差，使用预训练模型时应保持其原有预处理约定，这不能简单归因于网络深浅

固定均值、标准差的变换在尺度非零时可以逆转，本身不必然丢失信息；如果每张图单独减去自己的均值，则可能去掉绝对亮度等任务所需的信息，所以还要看具体如何处理

## Weight Initialization

对于初始化权重，一个简单的想法是随机设置一些数字，比如从均值为 0、方差为 $\sigma^2$ 的高斯分布 $\mathcal N(0,\sigma^2)$ 采样，随机性还可以打破不同神经元之间的对称，让它们学到不同特征

```python
# 6 层网络的前向传播，隐藏层宽度为 4096
dims = [4096] * 7
hs = []
x = np.random.randn(16, dims[0])
for Din, Dout in zip(dims[:-1], dims[1:]):
    W = 0.01 * np.random.randn(Din, Dout) # 随机初始化权重
    x = np.tanh(x.dot(W)) # 使用 tanh 激活
    hs.append(x)
```

对于小型网络还好，但是对于大型的网络问题很明显

当系数较小，比如 0.01，随着层数增加，后面各层的激活值可能逐渐往 0 聚集，反向传播的梯度也可能变小

![[CV导论/imgs/img5/image-3.png]]

当系数较大，比如 0.05，激活值又可能往 tanh 的两端聚集，进入导数很小的饱和区

![[CV导论/imgs/img5/image-4.png]]

两种情况都可能使训练困难，所以初始化不能随便指定同一个尺度，需要结合每层的输入数量和激活函数

### Xavier Initialization

Xavier 初始化的目的是

-  保持每一层的输入和输出的方差在前向传播中尽可能一致，避免信号在网络中逐层衰减或放大
-  尽量控制反向传播时的梯度尺度，减轻梯度爆炸或消失
为此，需要根据这一层参与计算的输入、输出数量来设置权重标准差 $std$

设一个输出为 $y=\sum_{i=1}^{D_{in}}x_iw_i$，$D_{in}$ 是参与这个输出计算的输入数量，记 $\operatorname{Var}$ 为方差

在各输入、权重相互独立，均值近似为 0，且输入、权重各自具有相同方差的近似下，希望 $\operatorname{Var}(y)$ 与 $\operatorname{Var}(x_i)$ 差不多，就得到下面的关系

$$
\begin{aligned}
\text{Var}(y) &= \text{Var}(x_1 w_1 + x_2 w_2 + \dots + x_{Din} w_{Din}) \\
 &= Din \ \text{Var}(x_i w_i) \\
 &= Din \ \text{Var}(x_i) \ \text{Var}(w_i)
\end{aligned}
$$

令两者方差相同，得到 $\operatorname{Var}(w_i)=1/D_{in}$，即权重标准差为 $1/\sqrt{D_{in}}$，这是只考虑前向传播的 fan-in 形式，标准 Xavier 初始化还会兼顾输入、输出数量，常取方差 $2/(D_{in}+D_{out})$

```python
W = np.random.randn(Din, Dout) / np.sqrt(Din)
```

对于大小为 $k\times k$ 的卷积，每个输出位置、每个输出通道会对 $k^2C_{in}$ 个输入加权求和，所以 $D_{in}=k^2C_{in}$

![[CV导论/imgs/img5/image-5.png]]

但实际上，更多情况下用的是 ReLU 而非 tanh，这时候最终又会趋向于 0

![[CV导论/imgs/img5/image-6.png]]

### He/Kaiming Initialization

对于近似对称、零均值的输入，ReLU 会把负半轴置零，使二阶矩大致减半，为了补偿这种信号衰减，He 初始化取权重方差 $2/D_{in}$，标准差则为 $\sqrt{2/D_{in}}$

```python
W = np.random.randn(Din, Dout) * np.sqrt(2/Din)
```

## Optimization

### Optimizer

对于 SGD，有一些问题

$$
x_{t+1} = x_t - \alpha \nabla f(x_t)
$$

当 loss 在一个方向变化剧烈，而在另一个方向变化较小，可能需要来回折返才最终达到最低点

![[CV导论/imgs/img5/image-7.png]]

同时，鞍点附近的梯度可能很小，导致更新缓慢，鞍点指沿一些方向是上坡、沿另一些方向是下坡的位置，高维情况下尤其需要考虑

此外，SGD 的梯度根据 mini-batch 的数据得出，噪声较大

#### SGD + Momentum

为了避免上述问题，考虑引入“动量”来模拟惯性，即梯度下降时还会沿之前的方向移动一段

$$
\begin{aligned}
v_{t+1} &= \rho v_t + \nabla f(x_t) \\
x_{t+1} &= x_t - \alpha v_{t+1}
\end{aligned}
$$

#### Adam

Adam 会结合梯度的历史方向和幅度，给不同参数调整更新尺度，是一种常见的优化器选择，但学习率仍需要结合任务设置

```python
# 初始化动量
first_moment = 0
second_moment = 0

# 训练迭代
for t in range(1, num_iterations):
    # 计算当前梯度
    dx = compute_gradient(x)  

    # 计算一阶动量和二阶动量
    first_moment = beta1 * first_moment + (1 - beta1) * dx  
    second_moment = beta2 * second_moment + (1 - beta2) * dx * dx

    # 偏差校正
    first_unbias = first_moment / (1 - beta1 ** t)
    second_unbias = second_moment / (1 - beta2 ** t)

    # 参数更新
    x -= learning_rate*first_unbias/(np.sqrt(second_unbias)+1e-7)
```

它结合了 **Momentum** 和 **AdaGrad / RMSProp** 的思想

1. 动量（Momentum）

```python
first_moment = beta1 * first_moment + (1 - beta1) * dx
```

一阶动量是梯度的指数加权平均，最近的梯度占一定权重，更早的梯度影响逐渐衰减
超参数 $\beta_1$ 衡量惯性，一般值为 $0.9$，即当前梯度占 $10\%$，历史梯度占 $90\%$

2. AdaGrad / RMSProp 的思想（指数加权平方平均）

```python
second_moment = beta2 * second_moment + (1 - beta2) * dx * dx
```

这段代码计算梯度的二阶动量（即梯度平方的加权平均），称为二阶动量估计
二阶动量会捕捉梯度的幅度（幅值），从而平衡不同方向的梯度更新，使我们在参数变化剧烈时减小更新幅度，变化平缓时增加更新幅度
超参数 $\beta_2$ 衡量“过往梯度平方的影响力”，一般设置为 $0.999$

3. 偏差校正（Bias Correction）

```python
first_unbias = first_moment / (1 - beta1 ** t)
second_unbias = second_moment / (1 - beta2 ** t)
```

两个动量都从 0 开始，早期会受到初值的影响，偏向 0，因此分别除以 $1-\beta_1^t$ 和 $1-\beta_2^t$，校正这种初始化偏差

4. 参数更新（结合两种动量）

公式（代码最后一行）：

```python
x -= learning_rate * first_unbias / (np.sqrt(second_unbias)+1e-7)
```

- 一阶动量（梯度方向）：指引优化的主要方向
- 二阶动量（学习率调整）：缩放梯度变化幅度，避免过大的步长
- $1e-7$ 是一个非常小的平滑项，防止除数为 0 导致的数值不稳定

### Learning Rate

学习率过小会训练缓慢，过大可能让损失震荡甚至发散，合适的范围与优化器、模型和任务有关，比如 Adam 中常会从 $10^{-3}$ 等量级尝试，再根据训练曲线调整

![[CV导论/imgs/img5/image-8.png|308x277]]

#### Iteration and Epoch

**Iteration 迭代**
指模型在训练中，对一个批次（batch）的数据更新一次参数的过程
通常包含前向传播、反向传播和一次优化器更新

**Epoch 训练周期**
一般指模型使用整个训练集（所有样本）完成一次训练的过程，包含多次迭代
训练时可以按 epoch 或一定的 iteration 间隔保存模型、记录曲线和评估验证集

实际上来讲，模型比较大时，可以在一个 epoch 内多保存几次模型
或者重新定义 epoch （比如多少个 iteration 算一个 epoch），此时 Learning Rate Schedule 中的 epoch 数需要调整（本质上是根据 iteration 数量）
一般来讲，batch size 与显存大小有关，假设是不变的

假设 batch size 不变，训练数据量翻倍，那么一个 epoch 中的 iteration 数也大致翻倍，比如从 $M$ 次变成 $2M$ 次，但对于扩大后的数据集，这仍然是 1 个 epoch，只是更新次数相当于原来的 2 个 epoch

而对于不同 batch size，假设总数据量一定，那 batch 的数量也要相应改变，此时训练效果不一定更好/更坏

#### Learning Rate Schedule

考虑在训练过程中动态地调整学习率，比如一开始较大，后来逐渐减小
可以在每 30/60/90 个 epochs 后让学习率 ×0.1，或者其它缩小方式

![[CV导论/imgs/img5/image-9.png|303x221]]

![[CV导论/imgs/img5/image-10.png]]

但是一上来就设置很高的学习率会让 loss 非常大，一开始可以线性增加学习率
这个过程称为 **Linear Warmup**

![[CV导论/imgs/img5/image-11.png|288x216]]

在一些 SGD 训练设置中，将 batch size 扩大 $N$ 倍时，会尝试将学习率也扩大 $N$ 倍，并配合 warmup，这是一种经验规则，仍需要观察训练是否稳定

# Underfitting & Overfitting

![[CV导论/imgs/img5/image-12.png]]

**欠拟合 underfitting**：模型连训练集都没有学好，可能是表达能力不足，也可能是优化过程没有找到合适参数

表达能力不足时，可以考虑增加深度或宽度；优化困难时，可以考虑归一化、残差连接等方法，先区分是哪类问题，再决定怎么调整

## Underfitting

### Batch Normalization

前面是在输入数据上做标准化，**批归一化 batch normalization (BN)**则把类似的处理放到网络内部，帮助控制中间特征的数值尺度，通常可以让优化更稳定

BN 最初以减轻中间特征的分布变化 internal covariate shift 为动机，它的优化收益还涉及其他机制，减均值、除标准差约束的是均值与方差，不要求特征服从高斯分布

结构：LBR (Linear - Batch Norm - ReLU)

![[CV导论/imgs/img5/image-13.png]]

训练阶段

- 输入：$x$，形状为 N×D（N 是批次大小，D 是特征维度）

1. 计算均值 $μ_j$

$$
\mu_j = \frac{1}{N} \sum_{i=1}^{N} x_{i,j}
$$

对批量中的每个通道（特征维度）逐列计算均值，结果形状为 D

2. 计算方差 $σ_j^2$

$$
\sigma_j^2 = \frac{1}{N} \sum_{i=1}^{N} (x_{i,j} - \mu_j)^2
$$

逐通道计算批量中特征的方差，结果形状为 D

 3. 归一化数据，得到 x̂

$$
\hat{x}_{i,j} = \frac{x_{i,j} - \mu_j}{\sqrt{\sigma_j^2 + \epsilon}}
$$

通过均值和标准差将数据标准化，结果形状为 N×D
$\epsilon$ 为一个小量，防止分母为 0

4. 引入可学习参数 γ 和 β

$$
y_{i,j} = \gamma_j \hat{x}_{i,j} + \beta_j
$$

γ 和 β 是可学习参数，用于调整归一化数据的分布，形状为 D
标准化消除了原输入的缩放和偏移信息， γ β 用于恢复模型的表达能力
一般被初始化为 $\gamma = 1$ 和 $\beta = 0$

测试阶段

- 测试时通常使用训练中累积的均值、方差估计，让同一个输入的结果不依赖当前与哪些样本放在一起，也方便逐个输入进行推理

- 这组统计量通常在每个训练 batch 后用**移动平均 moving average**更新，并非将整个训练集重新计算一遍，$\rho$ 表示保留历史估计的比例

$$
\begin{aligned}
\mu_{\text{rms},j} \leftarrow \rho \mu_{\text{rms},j} + (1 - \rho) \mu_{j} \\
\sigma_{\text{rms},j}^2 \leftarrow \rho \sigma_{\text{rms},j}^2 + (1 - \rho) \sigma_{j}^2
\end{aligned}
$$

- 测试时的归一化公式：

$$
y_{i,j} = \gamma_j \frac{x_{i,j} - \mu_{\text{rms},j}}{\sqrt{\sigma_{\text{rms},j}^2 + \epsilon}} + \beta_j
$$

对于 CNN ，将每一个通道视为一个特征维度
计算每个通道的所有样本与像素上的均值

$$
\begin{aligned}
\mu_c = \frac{1}{N \cdot H \cdot W} \sum_{n=1}^{N} \sum_{h=1}^{H} \sum_{w=1}^{W} x_{n,c,h,w} \\
\sigma_c^2 = \frac{1}{N \cdot H \cdot W} \sum_{n=1}^{N} \sum_{h=1}^{H} \sum_{w=1}^{W} \left(x_{n,c,h,w} - \mu_c\right)^2
\end{aligned}
$$

优点

- 改善优化：有助于稳定中间特征尺度，通常能够加快收敛
- 增强鲁棒性：对超参数（学习率、初始化等）较不敏感
- 提升泛化能力：对每个小批量数据的归一化具有一定的正则化效果

缺点

- 如果训练时 batch size 过小，$\mu$ 和 $\sigma$ 会很随机，噪声大，训练效果差
- 对训练集和测试集上的同一数据表现不一致（和同一个 batch 中的其它数据有关），会导致很多问题
- 如果移动统计量不准，或者训练、测试数据分布差异明显，测试模式下的表现可能变差

如果改用每个样本自身的特征来计算统计量，就不需要依赖同一个 batch 中的其他样本，Layer Norm、Group Norm 等采用了不同的归一化维度

![[imgs/img5/image-14.png]]

Batch Norm 对比 Group Norm

![[imgs/img5/image-15.png]]

当 batch size 过小时，可以用 Layer Norm / Group Norm 替代 BN

>Transformer 中常见的 Layer Norm，是对每个样本、每个 token 自己的特征维度求均值和方差，不跨其他 token，也不跨其他句子；BN 则对同一通道跨 batch，并在 CNN 中同时跨空间位置计算统计量
