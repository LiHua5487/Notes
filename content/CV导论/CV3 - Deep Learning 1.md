---
date: 2025-03-12
tags:
  - CV
---
# Equivariance and Invariance

我们希望 corner detector 能对图片的 translation 平移和 rotation 旋转都能正确应对，即在图片变化时， corner 的相对位置不变，但是绝对位置随着变化而改变，这种性质称为 **equivariance** 等变性

![[imgs/img3/image.png|665x268]]

其中 $f$ 是一些抽象操作（如卷积）
$T$ 是对图片的一些具体操作（如平移旋转）

$T[f(X)] = f(T(X))$ 即先平移旋转再处理，和先处理再平移旋转结果是一样的

在 edge corner 等位置性的检测中，希望等变的；
在分类任务中，则希望是不变的

要看 corner detector 是否等变，即看 response function $\theta$ 是否等变
而求梯度、平方是等变的，也就需要看卷积是否等变
- 对于平移，这是显而易见的，因为卷积就是滑动看这个局部的
	前提是对于不同局部用的是同一个卷积核
- 对于旋转，假设 image 是一个连续的东西（二维标量场），即分辨率无穷大
	如果卷积核是高斯核，由于高斯核是旋转不变的，这个局部的结果就是不变的，而这个局部的位置会随着旋转改变，那么整个就是等变的

但实际上
1. 图片是离散成pixel的，对于一个旋转的pixel，四个角会落到旁边四个pixel，对这四个pixel做双线性插值，这意味着对于旋转并不是严格等变的
2. 在实现高斯核时，只是针对 90° 180° 270° 旋转不变
这两点就导致 corner detector只是近似旋转等变
当分辨率越高时旋转等变性越好

另外，可以证明，对于图片的缩放是不等变的，corner 可能被认为是 edge

# Machine Learning

在之前的 detector中，所有参数都是手动设置的，我们希望能自动学习参数
即从 non-learning based method 到 learning based method
## Feature Extraction

- A feature is any piece of information which is relevant for solving the computational task related to a certain application.

Example

![[imgs/img3/image-1.png|486x223]]

while some less important features need to be abandoned.

可以根据这些搞一个 model，再根据输出结果评价，反过来调整 model

![[imgs/img3/image-2.png|298x206]]

假设我们有充足的数据，就可以调整出合适的参数 $\theta$ 的值 
对于神经网络，类比直线拟合，我们可以采用函数 $y = h_\theta(x)$ 来拟合
但是对于直线拟合，我们事先知道直线的函数形式，对于神经网络则不同

## Single-Layer Network

以手写数字识别为例，考虑一个简单的情况，判断一个数字是不是5

![[imgs/img3/image-3.png]]

use MNIST data base, with 28×28 resolution for each image
flatten the image to a one-dimensional vector $x \in R^{784}$ as input
Let’s assume a linear function $$h(x)=g(\theta^Tx)$$
$g(z)$ is a **Sigmoid Function**, which converts $z=\theta^Tx$ from $(-\infty,\infty)$ to $(0,1)$
$$g(z) = \frac{1}{1 + e^{-z}}$$
then we need to decide the fitting/training objective by using **Loss Function**

## Max Likelihood Estimation (MLE)

假设输入数据为 $x$ ，给定一个参数化概率模型 $p(y|\theta)$，其中：
- $y$ 是期望得到的观测结果；
- $\theta$ 是模型参数；
- $p(y|x;\theta)$ 表示在参数 $\theta$ 下，输入 $x$ 生成 $y$ 的概率（或概率密度）。

最大似然估计 MLE 寻找参数 $\theta$ 的值，使得观测到的数据 $\{y1​,y2​,…,yn​\}$ 的联合概率 $p(y1​,y2​,…,yn​|x;\theta)$ 最大，即什么情况下，最终结果长成这样的概率最大

对于每个数据组，可以得到
$$
\begin{align}
p(y = 1 \mid x; \theta) = h_\theta(x) \\
p(y = 0 \mid x; \theta) = 1 - h_\theta(x)
\end{align}$$
将 $y =0$ 和 $y=1$ 的情况写在一起，就变成
$$p(y \mid x; \theta) = \big(h_\theta(x)\big)^y \big(1 - h_\theta(x)\big)^{1-y}$$
假设不同数据组之间相互独立，那整体概率就变成
$$
\begin{align}
p(Y \mid X; \theta) &= \prod_{i=1}^n p\left(y^{(i)} \mid x^{(i)}; \theta\right) = \prod_{i=1}^n \left(h_\theta(x^{(i)})\right)^{y^{(i)}} \left(1 - h_\theta(x^{(i)})\right)^{1 - y^{(i)}} \\
\log p(Y \mid X; \theta) &= \sum_{i=1}^n y^{(i)} \log\left(h_\theta(x^{(i)})\right) + \left(1 - y^{(i)}\right) \log\left(1 - h_\theta(x^{(i)})\right)
\end{align}
$$
由此得到 **Negative Likelihood Loss (NLL)**
$$
\begin{align}
\mathcal{L}(\theta) 
&= - \log p(Y \mid X; \theta) \\
&= - \sum_{i=1}^n \left[ y^{(i)} \log\left(h_\theta(x^{(i)})\right) 
+ \left(1 - y^{(i)}\right) \log\left(1 - h_\theta(x^{(i)})\right) \right]
\end{align}
$$
而后最小化这个损失

## Gradient Decent

assume the Loss Function looks like this

![[imgs/img3/image-4.png|437x356]]

there are many **Local Minima**, but we wanna find the **Global Minima**
notice that the gradient of Loss implies the fastest direction to raise Loss
then we just need to move along the direction opposite to the gradient
so we can update the params like this
$$\theta := \theta - \alpha \nabla_\theta \mathcal{L}(\theta)$$where $\alpha$ is a hyper parameter, which means **Learning Rate**
- If $\alpha$ is small enough, then GD will definitely lead to a smaller loss after the update. However, a too small needs too many iterations to get the bottom.
- If is too big,  overshoot! Loss not necessary to decrease.

In the example above, for sigmoid function, we have
$$
\begin{align*}
g'(z) &= \frac{d}{dz} \frac{1}{1 + e^{-z}} \\
&= \frac{1}{(1 + e^{-z})^2} \cdot e^{-z} \\
&= \frac{1}{(1 + e^{-z})} \cdot \left(1 - \frac{1}{(1 + e^{-z})}\right) \\
&= g(z)(1 - g(z)).
\end{align*}
$$
for NLL, we have
$$
\begin{align*}
\mathcal{L} &= -\sum_{i=1}^{n} \left[ y^{(i)} \log(h_\theta(x^{(i)})) + (1 - y^{(i)}) \log(1 - h_\theta(x^{(i)})) \right] \\
\frac{\partial \mathcal{L}}{\partial \theta_j} &= -\sum \left( \frac{y}{g(\theta^T x)} - \frac{(1-y)}{1 - g(\theta^T x)} \right) \frac{\partial}{\partial \theta_j} g(\theta^T x) \\
&= -\sum \left( \frac{y}{g(\theta^T x)} - \frac{(1-y)}{1 - g(\theta^T x)} \right) g(\theta^T x) (1 - g(\theta^T x)) \frac{\partial}{\partial \theta_j} \theta^T x \\
&= -\sum \left( y(1 - g(\theta^T x)) - (1 - y)g(\theta^T x) \right) x_j \\
&= -\sum \left( y - h_\theta(x) \right) x_j.
\end{align*}
$$

If we take all data and label pairs in the training set to calculate the gradient, which is called **Batch Gradient Descent**, we may easily get trapped at local minima, and it's very slow.
Instead, we randomly sample N pairs as a batch from the training data and then compute the average gradient from them, which is called **Stochastic Gradient Descent (SGD)** , we have the potential to get out of local minima, and it's faster.

for SGD, the formula looks like this
$$
\nabla_W L(W) = \frac{1}{N} \sum_{i=1}^{N} \nabla_W L_i(x_i, y_i, W)
$$

For convex function, the local minima and its global minima are the same. 
the process of optimizing convex function is called **Convex Optimization**





