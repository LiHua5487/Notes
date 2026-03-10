
# Background

![[AI编程/imgs/img17/image.png]]

![[AI编程/imgs/img17/image-1.png]]

>一共有 32 位，最多只能表示 $2^{32}$ 个数，我们只是在调整到底能表示哪些数，Exponent 与 Fraction 的位数配比就代表了取值大小的范围和精度的权衡，上图这种分配方式称为 E8M23 

下图进一步划分了更多的情况

![[AI编程/imgs/img17/image-2.png]]

>normal 区域的精度受 Fraction 位数的影响；subnormal 区域的数受 Exponent 和 Fraction 的位数共同影响 

可表示的数的分布如下，我们希望对于更小的数字，有更高的精度，而对于很大的数字，小数点后几位就不重要了

![[AI编程/imgs/img17/image-3.png]]

# Quantization

对于一个高精度版本的模型，我们想让它在一些轻量化设备上也能用，量化的目标就是把模型权重用更低的精度存储下来，同时尽可能保持准确性

## K-means Quantization

不改变存储精度，而是减少存储的数字，设待存储的数据为 W ，过程如下
- 人为设定 K 个数作为初始“中心”
- 把 W 中的每个数分配给离它最近的中心点
- 重新计算每个组里所有点的平均值，将其作为新的中心点
- 根据新的中心的重新分组，直到分组情况不再变化

![[AI编程/imgs/img17/image-4.png]]

我们只需要存这 K 个数（codebook，保持原精度）和分组情况（indexes，一堆整数）

数学上，这个过程相当于最小化存储结果相较于原始数据的误差

$$
J = \sum_{i=1}^K \sum_{x \in C_i} \|x - \mu_i\|^2
$$

- $C_i$ 是第 $i$ 个组，$\mu_i$ 是该组的中心

---

采用这种存储方式，在更新存储的权重时，按照权重的分组情况，对梯度值进行同样的分组，并计算每组的平均值，拿这个平均值更新每组的中心值即可

![[AI编程/imgs/img17/image-5.png]]

## Linear Mapping

想要降低存储精度，但是低精度的表示范围可能和原精度不一样，考虑采用线性映射的方式对应过去

$$r=S\cdot(q-Z)$$

- $r$ 是原精度下的一个数，$q$ 是对应的低精度版本
- $S$ 是两种精度表示范围的比例，$Z$ 是低精度下，原精度中的 $0$ 对应的数，具体推导如下

![[AI编程/imgs/img17/image-6.png]]

---

采用这种存储方式，在计算矩阵乘法时，可以不必进行原精度下的计算，而是直接算出乘法结果对应的存储值

比如要计算 $Y = WX$ ，在低精度下，相当于

$$
S_Y (q_Y - Z_Y) = S_W (q_W - Z_W) \cdot S_X (q_X - Z_X)
$$

我们关注的是 $Y$ 的低精度版本 $q_Y$ 

$$
\begin{aligned}
q_Y &= \frac{S_W S_X}{S_Y} (q_W - Z_W)(q_X - Z_X) + Z_Y\\
&= \frac{S_W S_X}{S_Y} (q_W q_X - Z_W q_X - Z_X q_W + Z_W Z_X) + Z_Y
\end{aligned}
$$

在神经网络训练时，由于有各种中心化手段，其权重一般会以 0 为中心，则 $Z_W=0$ ，上式可简化为

$$
q_Y = \frac{S_W S_X}{S_Y} (q_W q_X - Z_X q_W) + Z_Y
$$

而 $Z_X q_W$ 这部分是可以预先计算的，因为它和具体的输入值 $X$ 无关

对于存在偏置的情况，也是类似的

![[AI编程/imgs/img17/image-7.png]]

>这里 N-bit 是转换成的低精度，32-bit 用于中间计算

## Quantization Granularity

实际上，神经网络中的权重是有很多通道的，对每个通道要分别量化

![[AI编程/imgs/img17/image-8.png]]

## Quantization during Training Pipeline

在反传过程中，量化这一步通常是导数为 0 的（因为可能会把一些高精度数字映射为同一个低精度数字，函数呈现阶梯型），为了避免梯度消失，将这一过程的梯度值视为 1 

![[AI编程/imgs/img17/image-9.png]]








