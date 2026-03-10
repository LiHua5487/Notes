---
date: 2025-03-19
tags:
  - CV
---

**Outline**

- Set up the task 
- Prepare the data $\rightarrow$ Need a labeled dataset. 
- Built a model $\rightarrow$ construct your neural network 
- Decide the fitting/training objective $\rightarrow$ Loss function 
- Perform fitting $\rightarrow$ Training by running optimization 
- Testing $\rightarrow$ Evaluating on test data

# Multilayer Perceptron (MLP)

There are some problem with Single-Layer Network.
One is that it can only handle linear separable cases.

$single\ layer\ z=\theta^Tx\ and\ sigmoid\ function\ g(z) \rightarrow g(\theta^Tx)$
$g(\theta^Tx)$ is a hyperplane in the space of $x$

![[imgs/img4/image.png|383x195]]

This model is lack of capacity and expressivity.
So we need **Multilayer Perceptron (MLP)**

## Fully-connected Layer

![[imgs/img4/image-1.png|618x284]]

MLP: Stacking linear layer and nonlinear activations

Q: why cannot simply stacking linear layer / why we need nonlinear activation
A: this is still a linear process, so the stacked layers is equivalent to one.
Similarly, simple stacking nonlinear layers is also not ok.

Through many non-linear layers, we can transform a linear non separable problem to linear separable at the last layer.

![[imgs/img4/image-2.png]]

say we have a 2 layer MLP

![[imgs/img4/image-3.png]]

extract features layer by layer, so the dimension is non-incremental.

![[imgs/img4/image-4.png]]

How can we acquire/update the parameters/weights of the MLP ?
## Backpropagation

![[imgs/img4/image-5.png]]

we can calculate $\frac{\partial q}{\partial x}$ and $\frac{\partial q}{\partial y}$ in advance after we calculated q.
use chain rule to calculate $\frac{\partial f}{\partial x}$ and $\frac{\partial f}{\partial y}$

![[imgs/img4/image-6.png]]

some common backpropagation operations

![[imgs/img4/image-6-1.png]]

The backpropagation can be efficiently implemented with simple matrix operations.

## Activate Functions

![[imgs/img4/image-7.png]]

Problem: sigmoid and tanh have Saturation zone 饱和区（导数近似0）
ReLU (Rectified Linear Unit) is a good default choice for most cases.

Above all, MLP model looks like this

![[imgs/img4/image-8.png]]

FC (fully connect layer) 一般用于高密集的信息处理
FC 层 = Linear + activation

Problems with flattening operation
- would be very expensive for high resolution images 
- breaks the local structure of an image

简而言之，直接将输入图片展平很不符合直觉与人类的感知方式


# Convolutional Neural Network (CNN)

![[imgs/img4/image-9.png]]
## Convolution Layer

Say we have a 32×32 image input with 3 channels (RGB)

![[imgs/img4/image-10.png]]

滑动 filter 窗口，直接将 filter 与 image 对应位置的元素(3 channel)做点积，得到1个数据（因为只有1个 filter，若有 $C_2$ 个则得到 $C_2$ 个数据）
这与数学上的卷积是等价的，不用翻转

we can get a number with $w^Tx + b$

![[imgs/img4/image-11.png]]

实际上，可以有多个 filter ，每个 filter 可以用不同的 $w$

![[imgs/img4/image-12.png]]

Q: bias 怎么设置？不同 filter？不同位置？
A：一个 filter 一个 bias，如果每个格子一个 bias，那求和后相当于一个 bias
$$\sum (w_i^Tp_i + b_i) = \sum w_i^Tp_i + \sum b_i = \sum w_i^Tp_i + b$$
若image的不同位置设置不同 bias，会破坏平移等变性
bias 的作用，以 ReLU 为例，bias 可以控制是否通过 ReLU
$$ReLU(\sum w_i^Tp_i + b)$$

**Convolution Kernel**

dimension: $k×k×C_1×C_2$
k 为卷积核大小， $C_1$ 为输入通道数， $C_2$ 为输出通道数（也是卷积核个数）

stride 步长：每次卷积核移动几格
$$Output size = (N - F) / stride + 1$$
- N 为输入图像大小，F 为卷积核大小
不能整除怎么办？ 引入 padding 操作

![[imgs/img4/image-13.png]]

We get a **Same** output, which does not change the dimension of the input.
say the filter size is F×F, the we can zero-padding with $(F - 1) / 2$ （补几圈）

Example

![[imgs/img4/image-14.png|366x178]]

![[imgs/img4/image-15.png|520x83]]

param for parameter 参数
if we directly flatten the image, we may need 30M+ params

![[imgs/img4/image-16.png]]

## Pooling Layer

![[imgs/img4/image-17.png|454x304]]

filter size=2×2, stride=2：以步长为2，分割为若干个 2×2 区域
Max Pooling: 每个区域取最大
Average Pooling: 每个区域取均值，等价于 2×2 等 weight 的卷积
Sum Pooling: 每个区域求和

![[imgs/img4/image-18.png]]

Input Dimensions： W₁ × H₁ × C
2 × 2 Pooling Operation： W₂ = ⌈W₁ / 2⌉     H₂ = ⌈H₁ / 2⌉  
Number of Parameters：0

the Architecture of CNN

![[imgs/img4/image-19.png]]

## Comparison


![[imgs/img4/image-20.png]]

**Sparse Connectivity**
![[imgs/img4/image-21.png|458x232]]

**Parameter Sharing**
![[imgs/img4/image-22.png]]

**Expressivity**
FC is a super set of CNN (without sparse and parameter sharing constraints)
so the expressivity of FC is higher

**Problem of FC/MLP**
- FC needs too many parameters
- FC will change dramatically even if you just shift one row or one column
	same happens for a slight rotation
- A huge problem during network training or optimization
	Besides overfitting, may have low acc even in training cases

卷积层的参数共享机制解决了平移的等变性
池化层解决了小范围的旋转的不变性

CNN stacks Conv layer and pooling layer.
Conv layer + pooling layer is invariant with small translation and rotation.
Thus CNN is much easier to optimize than a FC for an image
Loss landscape 有多个差不多好的 local minima






































