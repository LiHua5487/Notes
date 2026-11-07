---
date: 2025-03-12
tags:
  - CV
---

# Equivariance and Invariance

如果一张图片平移或旋转了，我们希望角点检测器仍能找到对应的角点，而且检测位置也跟着移动，这种“输入怎么变，输出也相应地变”的性质称为**等变性 equivariance**

![[CV导论/imgs/img3/image.png|665x268]]

用 $f$ 表示检测或特征处理，用 $T$ 表示平移、旋转等变换，等变性可以写为

$$
T(f(X))=f(T(X))
$$

即先处理再移动结果，与先移动图像再处理，得到的结果相同，而分类任务中，我们希望猫的照片移动后仍被判断为猫，输出类别不变，这称为**不变性 invariance**，写为 $f(T(X))=f(X)$

对于角点检测，需要看整个响应函数是否具有对应的等变性

- 平移：使用相同卷积核处理不同位置时，局部图案平移后响应也相应平移，先忽略图像边界的影响
- 旋转：在连续图像中，圆对称的高斯窗口不会偏向某个方向，结合梯度矩阵的旋转关系，Harris 响应具有旋转等变性

但实际图像是离散网格，任意角度旋转通常需要插值，离散高斯核也不能完美保留连续情况下的性质，所以只能近似旋转等变，分辨率较高时通常更接近连续情况

图像缩放则不同，同一个局部在固定窗口里可能从角点变成边缘，因此普通 Harris 检测并不具备尺度等变性

# Machine Learning

前面的检测器里，卷积核、阈值等往往是手动设置的，如果任务变复杂，我们希望这些参数能从数据中自动学出来，这就从手工设计方法转向了**机器学习 machine learning**

## Feature Extraction

**特征 feature**：与当前任务有关的信息，比如识别一个物体时，轮廓、纹理可能有用，背景颜色则可能不重要，**特征提取 feature extraction**就是保留有用信息、减少无关信息的影响

![[CV导论/imgs/img3/image-1.png|486x223]]

把提取出的特征交给模型 model，模型输出预测，再根据预测与真实结果的差异调整参数

![[CV导论/imgs/img3/image-2.png|298x206]]

可以把模型写成 $h_\theta(x)$，其中 $x$ 是输入，$\theta$ 是可调整的参数，模型结构决定可以表达哪些函数，训练则是在这些函数中寻找适合当前数据的一种

## Single-Layer Network

以手写数字识别为例，先只判断一张图片是不是数字 5，这叫**二分类 binary classification**，使用 MNIST 数据集，每张灰度图大小为 $28\times28$

![[CV导论/imgs/img3/image-3.png]]

先把图像展平 flatten，即把 784 个像素按固定顺序排成一个向量 $x\in\mathbb R^{784}$，再计算加权和并经过 sigmoid

$$
\begin{aligned}
z&=\theta^Tx \\
h_\theta(x)&=g(z)=\frac{1}{1+e^{-z}}
\end{aligned}
$$

$\theta^Tx$ 就是对应元素相乘后求和，$\theta$ 中每个权重表示对应输入如何影响结果，若需要偏置 $b$，可以写成 $\theta^Tx+b$，也可以在 $x$ 中添加一个恒为 1 的分量，把偏置并入 $\theta$

sigmoid 将任意实数压到 $(0,1)$，可以把输出解释为“这张图是 5”的概率，但模型一开始并不知道哪些权重合适，还需要定义**损失函数 loss function**，衡量预测有多差

## Maximum Likelihood Estimation (MLE)

设真实标签为 $y$，是 5 时 $y=1$，否则 $y=0$，用刚才的模型定义条件概率

$$
\begin{aligned}
p(y=1\mid x;\theta)&=h_\theta(x) \\
p(y=0\mid x;\theta)&=1-h_\theta(x) \\
p(y\mid x;\theta)&=h_\theta(x)^y\big(1-h_\theta(x)\big)^{1-y}
\end{aligned}
$$

最后一行只是把两种情况合在一起，代入 $y=0$ 或 $y=1$ 就能还原，**最大似然估计 MLE**希望调整 $\theta$，让模型给实际观测到的标签尽可能高的概率

假设各个样本独立，对 $n$ 组输入和标签，联合概率就是各组概率的乘积，取对数后，乘积变成求和，计算和优化都更方便

$$
\begin{aligned}
p(Y\mid X;\theta)&=\prod_{i=1}^n p(y^{(i)}\mid x^{(i)};\theta) \\
\log p(Y\mid X;\theta)
&=\sum_{i=1}^n\left[y^{(i)}\log h_\theta(x^{(i)})+(1-y^{(i)})\log\big(1-h_\theta(x^{(i)})\big)\right]
\end{aligned}
$$

取负号，把“越大越好”的目标转成“越小越好”，得到**负对数似然损失 negative log-likelihood loss (NLL)**，这里也就是二元交叉熵

$$
\mathcal L(\theta)=-\sum_{i=1}^n\left[y^{(i)}\log h_\theta(x^{(i)})+(1-y^{(i)})\log\big(1-h_\theta(x^{(i)})\big)\right]
$$

比如真实标签是 1，模型却只给它 0.01 的概率，损失就很大；如果给到 0.99，损失就很小，因此最小化这个损失会推动预测接近真实标签

## Gradient Descent

损失是参数的函数，训练就是找到能让损失较小的参数，可以把参数想成当前位置、损失想成地形高度，然后不断往下坡方向走

![[CV导论/imgs/img3/image-4.png|437x356]]

**梯度 gradient**由损失对各个参数的偏导数组成，指向局部增加最快的方向，因此沿其反方向更新，称为**梯度下降 gradient descent (GD)**

$$
\theta\leftarrow\theta-\alpha\nabla_\theta\mathcal L(\theta)
$$

$\alpha$ 称为**学习率 learning rate**，是人为设置的超参数，控制一次走多远，过小会训练缓慢，过大则可能越过低点、让损失增大，适当的小步长通常能降低当前损失，但一般的非凸目标不保证找到全局最小值

### Gradient of the Binary Classifier

先求 sigmoid 的导数

$$
\begin{aligned}
g'(z)&=\frac{e^{-z}}{(1+e^{-z})^2} \\
&=g(z)\big(1-g(z)\big)
\end{aligned}
$$

对一个样本，记 $h=g(\theta^Tx)$，根据链式法则，从损失对输出的影响，一步步乘回参数

$$
\begin{aligned}
\ell&=-y\log h-(1-y)\log(1-h) \\
\frac{\partial\ell}{\partial\theta_j}
&=\left(-\frac yh+\frac{1-y}{1-h}\right)h(1-h)x_j \\
&=(h-y)x_j \\
\frac{\partial\mathcal L}{\partial\theta_j}
&=\sum_{i=1}^n\big(h_\theta(x^{(i)})-y^{(i)}\big)x_j^{(i)}
\end{aligned}
$$

可以看到，梯度与预测误差有关，模型通过这个误差信息知道应该往哪个方向调整权重

### Batch and Mini-Batch

如果每次更新都计算整个训练集的梯度，称为 **batch gradient descent**，数据量大时，一次更新就很慢

因此可以随机选一小批数据 mini-batch，用这批数据的平均梯度近似整个训练集的平均梯度，严格说逐个样本更新才叫 SGD，不过实际也常把 mini-batch SGD 简称为 SGD

$$
\nabla_W L_{\text{batch}}(W)=\frac1N\sum_{i=1}^N\nabla_W L_i(x_i,y_i;W)
$$

这里 $N$ 是这批数据的大小，$W$ 也是模型参数的记号，小批量梯度带有噪声，每步成本较小，能更频繁更新，但并不保证每一步都降低整个训练集的损失

>凸函数的局部最小值也是全局最小值，前面的线性 sigmoid 二分类配合二元交叉熵，对参数是凸优化；复杂神经网络一般是非凸的，可能有多个局部最小值和鞍点，不能把它们混为同一种情况
