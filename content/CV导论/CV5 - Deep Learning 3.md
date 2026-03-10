---
date: 2025-03-26
tags:
  - CV
---
# CNN Training

Mini-batch SGD

Loop
- Sample a batch of data
	- dataloader 
	- shuffle (make sure it's random) and split the data into N batches
- Forward prop
- Backprop to calculate gradients
- Update parameters using gradients

## Data Preparation

![[imgs/img5/image.png]]

where X is a N×D data matrix, each example in a row

N : batch size D : channels

有时候，数据可能全正或全负，对于ReLU不适合，而且会影响梯度下降

![[imgs/img5/image-2.png]]


有时候，如果不归一化，分类结果会对权重变化很敏感

![[imgs/img5/image-1.png|700x353]]

在中心置零时，假设对于一堆RGB图像，要对每一个channel的所有图片的所有位置取mean，而不是一个图像的mean或者所有图像同一位置的mean

进行归一化有利于训练，但可能导致 information loss，比如说可能放大或缩小原来数据的差距，所以进不进行要视情况而定

- subtract per-channel mean (e.g. VGGNet)
- subtract per-channel mean + divide by per-channel std (e.g. ResNet)

这是因为 VGGNet 相对较浅，而 ResNet 更深，随着网络深度增加，模型对数据分布敏感性更强，归一化的优势远大于潜在的信息损失影响

## Weight Initialization

对于初始化权重，一个简单的想法是随机设置一些数字（如高斯分布 $N(0, \sigma^2)$）

``` python
# Forward pass for a 6-layer net with hidden size 4096
dims = [4096] * 7
hs = []
x = np.random.randn(16, dims[0])
for Din, Dout in zip(dims[:-1], dims[1:]):
    W = 0.01 * np.random.randn(Din, Dout) # set random weights
    x = np.tanh(x.dot(W)) # use tanh as activation function
    hs.append(x)
```

对于小型网络还好，但是对于大型的网络问题很明显

当系数较小，比如 0.01，后面 $\frac{\partial L}{\partial W}$ 会全都往 0 聚集

![[imgs/img5/image-3.png]]

当系数较大，比如 0.05，后面会全都往两边分开

![[imgs/img5/image-4.png]]

这都会导致模型几乎没有学习，所以不同 layer 应该区分对待

### Xavier Initialization

Xavier 初始化的目的是
-  保持每一层的输入和输出的方差在前向传播中尽可能一致，避免信号在网络中逐层衰减或放大
-  确保梯度在反向传播中不至于随着层数增加而爆炸或消失
为了达成这个目的，Xavier 初始化设计了适当的权重标准差 $std$，使得前向和反向传播时方差能够最佳地保持在合理范围内

Let: $y = x_1 w_1 + x_2 w_2 + ... + x_{Din} w_{Din}$
Assume: $\text{Var}(x_1) = \text{Var}(x_2) = \dots = \text{Var}(x_{Din})$
We want: $\text{Var}(y) = \text{Var}(x_i)$
$$
\begin{align}
\text{Var}(y) &= \text{Var}(x_1 w_1 + x_2 w_2 + \dots + x_{Din} w_{Din}) \\
 &= Din \ \text{Var}(x_i w_i) \\
 &= Din \ \text{Var}(x_i) \ \text{Var}(w_i)
\end{align}
$$
当 y 与 x 方差相同时，可得到 $Var(w_i)$ ，即 $std^2 = 1/Din$

``` python
W = np.random.randn(Din, Dout) / np.sqrt(Din)
```

对于卷积，假设输入 H×W×C_in，输出 H×W×C_out，卷积核大小为 k ,每一次卷积相当于对于 $k×k×C_{in}$ 个数转化为 $C_{out}$ 个数，所以 $Din = k×k×C_{in}$

![[imgs/img5/image-5.png]]

但实际上，更多情况下用的是 ReLU 而非 tanh，这时候最终又会趋向于 0

![[imgs/img5/image-6.png]]


### He/Kaiming Initialization

对于 ReLU，只有一半能通过，有效方差只有原来的一半
为了补偿这种方差减半的问题，需要更大的初始权重分布范围 $std^2 = 2/Din$

``` python
W = np.random.randn(Din, Dout) * np.sqrt(2/Din)
```

## Optimization

### optimizer

对于 SGD，有一些问题
$$
x_{t+1} = x_t - \alpha \nabla f(x_t)
$$
当 loss 在一个方向变化剧烈，而在另一个方向变化较小，可能需要来回折返才最终达到最低点

![[imgs/img5/image-7.png]]

同时，由于无法区分鞍点与极小值点，可能停在鞍点，而高维情况鞍点可能很多

此外，SGD 的梯度根据 mini-batch 的数据得出，噪声较大

#### SGD + Momentum

为了避免上述问题，考虑引入“动量”来模拟惯性，即梯度下降时还会沿之前的方向移动一段

$$
\begin{align}
v_{t+1} &= \rho v_t + \nabla f(x_t) \\
x_{t+1} &= x_t - \alpha v_{t+1}
\end{align}
$$
#### Adam

Adam is a good default choices working okay with constant learning rate

``` python
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

``` python
first_moment = beta1 * first_moment + (1 - beta1) * dx
```


这段代码是计算梯度的一阶动量（即梯度的加权平均），称为 一阶动量估计
超参数 $\beta_1$ 衡量惯性，一般值为 $0.9$，即当前梯度占 $10\%$，历史梯度占 $90\%$

2. AdaGrad / RMSProp 的思想（指数加权平方平均）

``` python
second_moment = beta2 * second_moment + (1 - beta2) * dx * dx
```

这段代码计算梯度的二阶动量（即梯度平方的加权平均），称为 二阶动量估计
二阶动量会捕捉梯度的幅度（幅值），从而平衡不同方向的梯度更新，使我们在参数变化剧烈时减小更新幅度，变化平缓时增加更新幅度。
超参数 $\beta_2$ 衡量“过往梯度平方的影响力”，一般设置为 $0.999$

3. 偏差校正（Bias Correction）

``` python
first_unbias = first_moment / (1 - beta1 ** t)
second_unbias = second_moment / (1 - beta2 ** t)
```

Adam 初始化时，动量值 $first\_moment$ 和 $second\_moment$ 都是从 0 开始，因此在早期阶段会有一些偏差——这些动量会比真实值偏低。
为了消除这一偏差，Adam 引入了一种 偏差校正机制
- 对动量进行除偏（去偏），使其更符合真实的梯度分布值。
- 加入 $(1 - \beta_1^t)$ 或 $(1 - \beta_2^t)$ 的修正因子，能够适应训练早期的梯度偏差问题

4. 参数更新（结合两种动量）

公式（代码最后一行）：

``` python
x -= learning_rate * first_unbias / (np.sqrt(second_unbias)+1e-7)
```

- 一阶动量（梯度方向）：指引优化的主要方向。
- 二阶动量（学习率调整）：缩放梯度变化幅度，避免过大的步长。
- $1e-7$ 是一个非常小的平滑项，防止除数为 0 导致的数值不稳定

### learning rate

对于学习率，过小或过大都会有问题，一个合适的范围是 1e-6 ~ 1e-3

![[imgs/img5/image-8.png|308x277]]

#### Iteration and Epoch

**Iteration 迭代** 
指模型在训练中，对一个批次（batch）的数据更新一次参数的过程。
包含前向传播，反向传播 / 梯度下降

**Epoch 训练周期** 
一般指模型使用整个训练集（所有样本）完成一次训练的过程，包含多次迭代
在一个训练周期后，会保存模型，绘制训练曲线，进行评估等

实际上来讲，模型比较大时，可以在一个 epoch 内多保存几次模型
或者重新定义 epoch （比如多少个 iteration 算一个 epoch），此时 Learning Rate Schedule 中的 epoch 数需要调整（本质上是根据 iteration 数量）
一般来讲，batch size 与显存大小有关，假设是不变的

假设训练数据量翻倍，那 batch 数量翻倍，假设原来为M个 iteration，则翻倍后变为2M个，不过这个2M相当于2个epoch，且比原来涵盖的数据更广

而对于不同batch size，假设总数据量一定，那 batch 的数量也要相应改变，此时训练效果不一定更好/更坏

#### Learning Rate Schedule

考虑在训练过程中动态地调整学习率，比如一开始较大，后来逐渐减小
可以在每 30/60/90 个 epochs 后让学习率 ×0.1，或者其它缩小方式

![[imgs/img5/image-9.png|303x221]]

![[imgs/img5/image-10.png]]


但是一上来就设置很高的学习率会让 loss 非常大，一开始可以线性增加学习率
这个过程称为 **Linear Warmup**

![[imgs/img5/image-11.png|288x216]]

一个经验性的结论是，如果将 batch size ×N，那也让初始学习率 ×N

# Underfitting & Overfitting

![[imgs/img5/image-12.png]]

Underfitting on the train set
- usually caused by limited model capacity or unsatisfactory optimization
- Batch normalization 
- Skip link

对于模型capacity不够，先考虑加深，再考虑加宽

## Underfitting
### Batch normalization

为什么需要 Batch Normalization
- 每次你在训练过程中更新模型的权重时，都会影响前一层向后一层传递的输出值的分布，换句话说，每一层的输入分布会因为参数更新而变化，接下来的层不得不去适应这种变化，即协方差漂移 **covariance shift**，这可能会导致训练变得困难
- Batch Norm 的目标是让网络的中间层（隐层）或输入数据的分布更加稳定（趋近均值为 0，方差为 1 的标准正态分布），从而加快训练并改善性能

结构：LBR (Linear - Batch Norm - ReLU)

![[imgs/img5/image-13.png]]

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

- 测试过程中，往往无法使用 batch 均值和方差，因为推理过程通常是逐个输入进行的，使用整个训练数据的全局均值和全局方差来进行归一化

- 这个全局均值和方差在每一次训练后更新，这个过程称为 **moving average** 
$$
\begin{align}
\mu_{\text{rms},j} \leftarrow \rho \mu_{\text{rms},j} + (1 - \rho) \mu_{j} \\
\sigma_{\text{rms},j}^2 \leftarrow \rho \sigma_{\text{rms},j}^2 + (1 - \rho) \sigma_{j}^2
\end{align}
$$

- 测试时的归一化公式：
$$
y_{i,j} = \gamma_j \frac{x_{i,j} - \mu_{\text{rms},j}}{\sqrt{\sigma_{\text{rms},j}^2 + \epsilon}} + \beta_j
$$

对于 CNN ，将每一个通道视为一个特征维度
计算每个通道的所有样本与像素上的均值
$$
\begin{align}
\mu_c = \frac{1}{N \cdot H \cdot W} \sum_{n=1}^{N} \sum_{h=1}^{H} \sum_{w=1}^{W} x_{n,c,h,w} \\
\sigma_c^2 = \frac{1}{N \cdot H \cdot W} \sum_{n=1}^{N} \sum_{h=1}^{H} \sum_{w=1}^{W} \left(x_{n,c,h,w} - \mu_c\right)^2
\end{align}
$$

优点
- 加速收敛：显著减少梯度消失和梯度爆炸问题
- 增强鲁棒性：对超参数（学习率、初始化等）较不敏感
- 提升泛化能力：对每个小批量数据的归一化具有一定的正则化效果

缺点
- 如果训练时 batch size 过小，$\mu$ 和 $\sigma$ 会很随机，噪声大，训练效果差
- 对训练集和测试集上的同一数据表现不一致（和同一个 batch 中的其它数据有关），会导致很多问题
- 测试时，对于训练数据表现也可能较差

不在batch维度上算mean，那train和test时候对于同样的数据表现就是一样的

![[imgs/img5/image-14.png]]

Batch Norm 对比 Group Norm

![[imgs/img5/image-15.png]]

当 batch size 过小时，可以用 Layer Norm / Group Norm 替代 BN