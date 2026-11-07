---
date: 2025-03-19
tags:
  - CV
---

训练一个模型，大致需要先明确任务、准备带标签的数据、选择模型结构、定义损失，再通过优化调整参数，最后用独立的测试数据评价效果

# Multilayer Perceptron (MLP)

前面的模型先计算 $z=\theta^Tx+b$，再经过 sigmoid，如果以 0.5 为分类阈值，分类边界就是 $\theta^Tx+b=0$，在二维平面中是一条直线，在更高维空间中称为超平面 hyperplane

![[CV导论/imgs/img4/image.png|383x195]]

但有些数据无法用一条直线或一个超平面分开，称为**线性不可分**，这时模型的表达能力不够，可以引入**多层感知机 multilayer perceptron (MLP)**

## Fully-Connected Layer

**全连接层 fully-connected layer (FC)**把输入向量的各个分量与每个输出连接起来，每个输出都有自己的权重和偏置，线性部分写为 $z=Wx+b$，再经过激活函数得到 $h=g(z)$

![[CV导论/imgs/img4/image-1.png|618x284]]

为什么不能直接把多个线性层叠起来？因为两个线性/仿射变换的复合仍然是同类变换

$$
W_2(W_1x+b_1)+b_2=(W_2W_1)x+(W_2b_1+b_2)
$$

所以需要在线性层之间插入**非线性激活函数 activation function**，这样多层组合才能表达更复杂的关系，把原来难分的数据逐步变成最后一层容易区分的特征

![[CV导论/imgs/img4/image-2.png]]

比如一个两层 MLP 可以先得到隐藏特征，再根据这些特征输出类别分数

![[CV导论/imgs/img4/image-3.png]]

![[CV导论/imgs/img4/image-4.png]]

每一层的输出维数由模型设计决定，可以增大也可以减小，并不要求逐层下降，那么这些层里的参数要怎么更新？

## Backpropagation

前向传播 forward propagation 按层计算预测和损失，**反向传播 backpropagation**则利用链式法则，从损失开始往回计算每个参数的梯度

![[CV导论/imgs/img4/image-5.png]]

比如 $q=xy$、$L=f(q)$，已知后面传来的 $\partial L/\partial q$，就可以计算

$$
\begin{aligned}
\frac{\partial L}{\partial x}&=\frac{\partial L}{\partial q}\frac{\partial q}{\partial x}=\frac{\partial L}{\partial q}y \\
\frac{\partial L}{\partial y}&=\frac{\partial L}{\partial q}\frac{\partial q}{\partial y}=\frac{\partial L}{\partial q}x
\end{aligned}
$$

每个计算单元只需要保存前向计算中必要的中间结果，并知道自己的局部导数，就能把梯度继续向前传；如果一个变量经过多条路径影响损失，各条路径传回的梯度需要相加

![[CV导论/imgs/img4/image-6.png]]

![[imgs/img4/image-6-1.png]]

实际可以利用矩阵运算高效完成，反向传播负责算梯度，之后再由梯度下降等优化器更新参数

## Activation Functions

![[imgs/img4/image-7.png]]

sigmoid 和 tanh 在输入绝对值较大时会进入**饱和区 saturation zone**，导数接近 0，梯度经过很多层连乘后可能变得很小，导致前面的层难以学习

一个常见选择是 **ReLU (Rectified Linear Unit)**，定义为 $\operatorname{ReLU}(z)=\max(0,z)$，正半轴的导数为 1，可以缓解这一问题，负半轴的导数则为 0

这样，一个 MLP 就由多次线性变换与非线性激活交替组成，严格说 FC 可以单指线性层，描述网络模块时也常把后面的激活一起画出来

![[imgs/img4/image-8.png]]

如果输入是高分辨率图像，直接展平后使用 FC，参数会非常多，而且没有利用像素之间的空间邻近关系，比如相邻像素与远处像素在连接方式上没有区别，这就引出了 CNN

# Convolutional Neural Network (CNN)

![[imgs/img4/image-9.png]]

## Convolutional Layer

假设输入是 $32\times32\times3$ 的 RGB 图像，一个大小为 $k\times k$ 的 filter 会同时覆盖输入的全部 3 个通道

![[imgs/img4/image-10.png]]

在一个位置上，把局部区域与 filter 的对应元素相乘并求和，再加一个偏置 $b$，得到一个数，写成展平后的形式就是 $w^Tx+b$，随着 filter 在图像上滑动，就得到一张**特征图 feature map**

![[imgs/img4/image-11.png]]

一个 filter 得到一个输出通道，使用多个 filter 就得到多个输出通道，不同 filter 可以学习检测不同的局部模式

![[imgs/img4/image-12.png]]

如果输入通道数为 $C_{in}$、输出通道数为 $C_{out}$，所有卷积核一共包含 $k^2C_{in}C_{out}$ 个权重，此外通常每个输出通道有一个偏置，即总参数量为 $k^2C_{in}C_{out}+C_{out}$

>这里按深度学习惯例称为卷积，实际计算通常是不翻转核的互相关；它与数学卷积的具体计算约定不同，但可学习卷积核可以相应翻转，表达同一类映射

### Parameter Sharing and Bias

同一个 filter 在所有空间位置使用相同权重和偏置，称为**参数共享 parameter sharing**，相当于不管局部图案出现在左上还是右下，都用同一种方式识别

如果在一个窗口内部给每项乘积都加偏置，最后求和时这些偏置可以合成一个数

$$
\sum_i(w_ix_i+b_i)=\sum_iw_ix_i+\sum_ib_i=w^Tx+b
$$

但如果给输出图的每个空间位置设置不同偏置，就会破坏这种统一处理位置的方式，偏置还会影响激活函数，例如在 ReLU 前移动加权和，使其更容易或更难大于 0

### Stride and Padding

**步长 stride**表示 filter 每次移动几格，**填充 padding**则是在图像边缘补上额外像素，常用补 0 的方式

对于输入一边长为 $N$、核大小为 $F$、两侧各补 $P$、步长为 $S$ 的普通卷积，输出边长为

$$
N_{out}=\left\lfloor\frac{N+2P-F}{S}\right\rfloor+1
$$

向下取整表示只计算能完整放下卷积核的位置，padding 既可以控制输出大小，也可以让边缘像素参与更多计算，并非只有“不能整除”时才使用

![[imgs/img4/image-13.png]]

当 $S=1$、$F$ 为奇数时，取 $P=(F-1)/2$ 就能保持宽高不变，称为 same 大小的输出；步长大于 1 时，需要结合具体输出大小确定 padding

![[imgs/img4/image-14.png|366x178]]

![[imgs/img4/image-15.png|520x83]]

图中的 param 是 parameter 的缩写，即参数，卷积的局部连接与参数共享可以显著减少参数量

![[imgs/img4/image-16.png]]

## Pooling Layer

**池化 pooling**把每个通道的局部区域压缩成一个数，通常用于减小空间分辨率，同时保留较显著或较稳定的局部信息

![[imgs/img4/image-17.png|454x304]]

比如大小为 $2\times2$、步长为 2 的池化，将图像划成不重叠的小块

- Max pooling：每块取最大值
- Average pooling：每块取平均值，相当于用固定权重做局部平均
- Sum pooling：每块求和

![[imgs/img4/image-18.png]]

如果输入宽高都是偶数，不补边时输出的宽高分别减半，通道数不变；若是奇数，需要看是否补边、是否启用向上取整，不能统一写成向上取整，普通池化也没有可学习参数

## Architecture and Comparison

一个常见 CNN 会交替使用卷积、激活与下采样，逐步提取特征，最后用全连接层或其他结构完成分类

![[imgs/img4/image-19.png]]

![[imgs/img4/image-20.png]]

**局部连接 sparse connectivity**：每个卷积输出只看上一层的一块局部区域，随着层数增加，间接看到的范围逐渐扩大

![[imgs/img4/image-21.png|458x232]]

**参数共享 parameter sharing**：同一个 filter 在各个位置复用，因此参数量不随图像宽高直接增长

![[imgs/img4/image-22.png]]

同样规模的全连接映射可以表示更一般的连接关系，而 CNN 把“局部模式重要、同一模式可能出现在不同位置”这些图像规律直接写进了结构，减少需要学习的自由度，往往更适合图像任务

步长为 1 的卷积在忽略边界时具有平移等变性，池化可以减弱小范围位置变化的影响，但下采样、边界处理会影响严格等变性，普通 CNN 也不自动保证旋转不变，分类上的稳定性还需要结合结构和训练数据
