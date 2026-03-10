
见 CV 导论 CV 3~7 ，下为补充部分

# Optimizer

## SGD / mini-batch SGD

如果每次都用整个训练集的所有数据去进行梯度更新，需要对每一个输入数据算一遍梯度，计算量太大了，所以将数据集划分为若干个 batch ，每次只计算一个 batch 上的平均梯度，去近似整个数据集上的平均梯度

## SGD + momentum

SGD 的问题
- 当 loss 在一个方向变化剧烈，而在另一个方向变化较小，可能需要来回折返才最终达到最低点
- 无法区分鞍点与极小值点，可能停在鞍点，而高维情况鞍点可能很多
- SGD 的梯度根据一个 batch 的数据得出，噪声较大

引入动量项，模拟”惯性“，这有两种表达形式

![[CV/img/img12-18/image-2.png]]

>这两种形式其实是等价的，因为一开始速度都是 0 ，相当于一个一直在累积负值，一个在一直累积正值

---

Nesterov Momentum ：先沿着速度方向移动一段距离，计算这一处的梯度（而不是用一开始的位置的梯度），然后再沿着梯度方向移动

![[CV/img/img12-18/image-1.png]]

但是实际计算时，我们反传时直接得到的是关于 $w_t$ 的梯度，而不是 $w_t+\rho v_t$ 处的梯度，所以考虑把 $w_t+\rho v_t$ 这个位置视为模型参数，定义

$$\widetilde{w}_t = w_t + \rho v_t$$

我们需要更新 $\widetilde{w}_t$ 到 $\widetilde{w}_{t+1}$ ，根据定义

$$
\widetilde{w}_{t+1} = w_{t+1} + \rho v_{t+1}
$$

考虑 Tensor Flow 的更新公式，为 $w_{t+1} = w_t + v_{t+1}$，代入得

$$
\widetilde{w}_{t+1} = (w_t + v_{t+1}) + \rho v_{t+1} = w_t + (1 + \rho)v_{t+1}
$$

又因为 $w_t = \widetilde{w}_t - \rho v_t$ ，则

$$
\widetilde{w}_{t+1} = (\widetilde{w}_t - \rho v_t) + (1 + \rho)v_{t+1} = \widetilde{w}_t - \rho v_t + (1 + \rho)v_{t+1}
$$

因此，我们得到等价更新公式

$$
\begin{aligned}
v_{t+1} &= \rho v_t - \alpha \nabla L(\widetilde{w}_t)\\
\widetilde{w}_{t+1} &= \widetilde{w}_t - \rho v_t + (1 + \rho)v_{t+1}
\end{aligned}
$$

```python
for t in range(num_steps):
    dw = gradient(w)
    old_v = v
    v = rho * v - learning_rate * dw
    w -= rho * old_v - (1 + rho) * v
```

## AdaGrad and RMSProp

神经网络中不同参数所需的更新幅度可能不同，如果所有参数都用同样的学习率，可能会导致学习过程不稳定或缓慢

AdaGrad 的核心思想是：根据每个参数的历史梯度的平方的累积和，来调整它的学习率
- 如果历史梯度很大，说明已经有很大变化了，需要放慢一些
- 如果历史梯度较小，说明没怎么动，可以试着走快一点


$$
\theta_i^{(t+1)} = \theta_i^{(t)} - \frac{\eta}{\sqrt{G_i^{(t)} + \epsilon}} \cdot \nabla J(\theta_i^{(t)})
$$

- $\eta$ 是初始学习率，是一个超参数
- $G_i^{(t)}$ 是参数 $\theta_i$ 过去所有梯度平方的累积和
- $\epsilon$ 是一个很小的数（比如 $10^{-8}$），防止除以零
- $\nabla J(\theta_i^{(t)})$ 是当前梯度

```python
grad_squared = 0 # 需要跨epoch存储
for t in range(num_steps):
    dw = gradient(w)
    grad_squared += dw ** 2
    w -= learning_rate * dw / (grad_squared.sqrt() + 1e-7)
```

但是有个问题，由于梯度平方是个正数， $G_i^{(t)}$ 会一直增加，导致学习率一直减小，可能过早的停止学习

---

RMSProp 对 AdaGrad 的问题进行了改进，在计算 $G_i^{(t)}$ 时，不直接把当前梯度的平方加上去，而是加权求和

$$
G_i^{(t)} = \beta \cdot G_i^{(t-1)} + (1 - \beta) \cdot g_t^2
$$

- $\beta$ 是梯度平方累积和的衰减率，是一个超参数
- 引入这个系数后，历史的梯度就会逐渐被”遗忘“，更多的关注最近的梯度

```python
grad_squared = 0
for t in range(num_steps):
    dw = gradient(w)
    grad_squared = beta * grad_squared + (1 - beta) * (dw ** 2)
    w -= learning_rate * dw / (grad_squared.sqrt() + 1e-7)
```

## Adam

Adam 结合了 momentum 和自适应学习率 RMSProp 的思想，既累积历史梯度（一阶矩），也累积历史梯度的平方（二阶矩）

1. 一阶矩的计算公式为

$$
m_t = \beta_1 \cdot m_{t-1} + (1 - \beta_1) \cdot g_t
$$

	- $m_t$ 是一阶矩，模拟动量
	- $\beta_1$ 是一阶矩的衰减率，通常设为 $0.9$ 
	- $g_t$ 是当前梯度

2. 二阶矩的计算公式为

$$
v_t = \beta_2 \cdot v_{t-1} + (1 - \beta_2) \cdot g_t^2
$$

	- $v_t$ 是二阶矩，与 RMSProp 相同
	- $\beta_2$ 是二阶矩的衰减率，通常设为 $0.999$ 

3. 而后需要对它们进行偏差矫正

$$
\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_1^t}
$$

4. 最后按照下面的公式更新参数

$$
\theta_{t+1} = \theta_{t} - \eta \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
$$

```python
m1 = 0
m2 = 0

for t in range(num_steps):
	dw = gradient(w)  
	
	m1 = beta1 * m1 + (1 - beta1) * dw
	m2 = beta2 * m2 + (1 - beta2) * (dw ** 2)
	
	m1_unbias = m1 / (1 - beta1 ** t)
	m2_unbias = m2 / (1 - beta2 ** t)
	
	w -= learning_rate * m1_unbias / (m2_unbias.sqrt() + 1e-7)
```

---

如果不进行偏差矫正，那一开始 m1 和 m2 就会非常接近 0 （因为 $1-\beta$ 会非常小），导致训练不稳定

那为啥是上面那样的矫正公式呢？以二阶矩为例，可见其更新公式是一个递推式

$$
v_t = \beta_2 \cdot v_{t-1} + (1 - \beta_2) \cdot g_t^2
$$

循环代入 $v_i$ 可得

$$
v_t = (1 - \beta_2) \sum_{i=1}^{t} \beta_2^{t-i} \cdot g_i^2
$$

考虑二阶矩的期望，并利用期望的线性性，可得

$$
\mathbb{E}[v_t] = \mathbb{E}\left[ (1 - \beta_2) \sum_{i=1}^{t} \beta_2^{t-i} \cdot g_i^2 \right] = (1 - \beta_2) \sum_{i=1}^{t} \beta_2^{t-i} \cdot \mathbb{E} \left[ g_i^2 \right]
$$

为简化问题，假设 $g_i$ 是独立同分布的，则 $\mathbb{E} \left[ g_i^2 \right]$ 均相同，即 $\mathbb{E} \left[ g_i^2 \right] = \mathbb{E} \left[ g_t^2 \right]$ ，那么上式就变为

$$
\mathbb{E}[v_t] = \mathbb{E} \left[ g_t^2 \right](1 - \beta_2) \sum_{i=1}^{t} \beta_2^{t-i} = \mathbb{E} \left[ g_t^2 \right] (1 - \beta_2^t)
$$

我们希望 $\mathbb{E}[v_t] = \mathbb{E} \left[ g_t^2 \right]$ ，所以需要除以 $1 - \beta_2^t$ 进行矫正（注意代码中将矫正结果设为一个新的变量，而不是直接修改原有变量，不然会破坏上面的递推式）

## AdamW

AdamW 在 Adam 的基础上加入了正则化 weight decay 避免过拟合

![[CV/img/img12-18/image.png]]

# Model Complexity

衡量一个模型的复杂度的指标
- 参数量：模型有多少可学习的参数 
- 浮点运算 FLOPs ：模型前向传播需要多少次运算  
	- 许多论文只计算卷积层的运算，忽略 ReLU、池化、Batch Norm 等  
	- 大多数论文采用 “1 FLOP = 1 次乘法和 1 次加法”，那么两个 N 维向量的点积需要 N FLOPs  
- 运行时间：模型在实际硬件上前向传播需要多长时间

# CNN Architecture

## AlexNet

![[CV/img/img12-18/image-3.png]]

>之所以分上下两部分，是因为一开始一个 GPU 放不下，只能分开放到两个 GPU 上面 

特点
- 更深的神经网络（相比传统的 LeNet ）
- 使用了 ReLU 激活函数（相比 LeNet 中的 tanh 和 sigmoid ）
- 局部响应归一化 LRN 
- 数据增强
- Dropout （在 FC 层后面）

> LRN ：在卷积层和池化层之间，对卷积层输出的每个特征图上的每个位置，计算该位置周围像素的平方和，然后将当前位置的像素值除以这个和；活跃度特别高的神经元（像素）会抑制其相邻神经元，在一定程度上能够避免过拟合，并提高网络的泛化能力

## VGGNet

![[CV/img/img12-18/image-4.png]]

用多层的 3×3 卷积来代替 5×5 11×11 这样的卷积，能用更少的参数达到相同的感受野

## GoogLeNet

>由 Google 团队造的，GoogLeNet 中的 L 大写是为了致敬 LeNet

1. 引入 inception 模块，多路并行，而且使用 1×1 卷积（将每个位置的不同通道的像素值合并，图像大小不变）降低通道维度

![[CV/img/img12-18/image-5.png]]

2. 最后分类时不是直接把一个很大的 FC 层接上去，而是先进行 global avg pooling （把 H×W 变为 1 ），再接一个不那么大的 FC 层 

![[CV/img/img12-18/image-6.png]]

3. 为了解决很深的神经网络在反传时的梯度消失问题，每隔一定层数设置一个辅助分类器，这样梯度可以直接从网络中间流进去（但是后来出现 Batch Norm 后，就不用这么干了，因为 BN 能让梯度更稳定）

![[CV/img/img12-18/image-7.png]]

## ResNet

![[CV/img/img12-18/image-8.png]]

引入 residual link ，很简洁的解决了反传时的梯度消失问题

## ResNeXt

![[CV/img/img12-18/image-11.png]]

将残差块中的 3×3 卷积改为分组卷积，把特征通道分成若干组（由一个超参数基数 cardinality 控制），减少了计算量，同时增加了模型表达的“多样性”

## DenseNet

![[CV/img/img12-18/image-9.png]]

把每两层之间都连起来，而且与 ResNet 不同，相连的两层的特征图会被拼到一起，而不是相加；后续的优化思路就变成了删去 / 保留哪些边的图优化问题

## SENet

![[CV/img/img12-18/image-10.png]]

在传统的 CNN 架构中，通常采用卷积层和池化层来提取图像特征，但这种方法并没有明确地对特征通道之间的关系进行建模

SE 模块引入 Squeeze 和 Excitation 操作来建模通道之间的关系，学习生成一个通道的权重向量，这个权重向量被应用于原始特征图上的每个通道，以对不同通道的特征进行加权，能够自适应地学习到每个通道的重要性






































