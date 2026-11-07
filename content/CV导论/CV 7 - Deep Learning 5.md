---
date: 2025-04-09
tags:
  - CV
---

# ImageNet Classification

ImageNet 分类任务需要根据整张图判断其中的主要物体，前面的卷积层提取特征，后面的分类部分输出各类别的分数

## VGGNet

![[CV导论/imgs/img7/image.png|330x383]]

相比 AlexNet，VGGNet 使用更深的网络和更小的卷积核，典型做法是多次堆叠 $3\times3$ 卷积

## Receptive Field

![[CV导论/imgs/img7/image-1.png]]

**感受野 receptive field**：一个特征位置能够受到输入中多大范围的影响，比如一层 $5\times5$ 卷积的一个输出，只直接依赖上一层的 $5\times5$ 区域；多层组合后，再把这种依赖追溯到原始输入

如果一直盯着小的局部看，那得到的信息量很有限，判断效果差
所以最终的感受野要尽量扩大到全图
每次卷积和池化都会让感受野增长，最终看到比较完整的特征
带步长的卷积或池化会增大相邻特征对应的原图间距，让后续层的感受野增长得更快，具体增长量与核大小、步长有关

![[CV导论/imgs/img7/image-2.png]]

假设做了 3 次步长为 1 的 $3\times3$ 卷积，且忽略边界
那 A3 在 A2，A1, Input 上的感受野分别为  3×3，5×5，7×7
如果直接在 Input 使用 7×7 的卷积核，那感受野也是 7×7
但实际上效果并不一样，因为3次卷积会有3次 ReLU，能力更强
如果各层通道数相同并暂不计偏置，3 次 $3\times3$ 的权重数量为 $27C^2$，一次 $7\times7$ 为 $49C^2$，因此前者参数更少，还能插入更多非线性激活

1 × 1 的卷积和 FC 并不一样，因为 FC 的感受野是前面整个图

## ResNet

![[CV导论/imgs/img7/image-3.png|631x533]]

基本结构

- 一开始进行卷积和池化
- 而后堆叠 residual block
- 每堆叠几次用一次步长为 2，channel 数更高的卷积
降低分辨率，但是提高了 channel 数，逐渐压缩提取特征
总深度主要由各阶段残差块的数量决定，下采样则控制空间分辨率

- 最后再 Global Pooling 和 FC

为了提高 efficiency ，采用以下结构

![[CV导论/imgs/img7/image-4.png|424x306]]

先用 1×1 卷积降低 channel 数，最后再用 1×1 卷积升回去
昂贵的 $3\times3$ 卷积在较少通道上计算，从而减少参数和计算量，前后 $1\times1$ 卷积负责混合、压缩与恢复通道，本身不会扩大空间感受野

假设输入大小为 $W\times W\times C$
这个结构形似漏斗，称为**瓶颈 bottleneck**，例如先将通道从 $C$ 压到 $C/4$，中间特征大小就是 $W\times W\times(C/4)$，再升回 $C$

![[CV导论/imgs/img7/image-5.png]]

# Segmentation

**图像分割 segmentation**需要判断图中各个像素属于什么，而分类只给整张图一个结果，分割还必须保留每个位置的信息

![[CV导论/imgs/img7/image-6.png]]

- Semantic segmentation：为每个像素标记类别，比如人、车、道路
- Instance segmentation：在标记类别的同时，区分同类物体中的不同个体，比如把两个人分开

granularity 颗粒度，颗粒度小，那划分的就越精细
grounding 视觉定位，提取物体的边界框

![[CV导论/imgs/img7/image-7.png]]

如果对于每一个小的局部的像素单独处理，显然是没法分辨的
如果直接对图片多次卷积提高感受野，再进行识别预测，那开销会很大
而且一直卷积可能丢失输入图片中的位置信息，比如边界或形状，那就难以恢复

![[CV导论/imgs/img7/image-9.png]]

## Auto Encoder

先把输入经过 Encoder 压缩，再经过 Decoder 恢复维度

![[CV导论/imgs/img7/image-8.png|275x169]]

中间的低维表示 $z$ 构成信息瓶颈 information bottleneck，希望通过降维去掉冗余信息，保留恢复输出所需的主要特征，分割网络也常使用这种 encoder-decoder 思路

如果输入中的信息量几乎没什么冗余，那直接压缩会造成信息丢失
但如果原来信息就存在大量冗余，那就可以压缩剔除
但是如果压缩太多，会出现不可逆转的信息丢失

比如对于一张试卷的照片，可以先提取出文字，再打印出来恢复

![[CV导论/imgs/img7/image-10.png]]

## Max Unpooling

max pooling 时可以记下每块最大值来自哪个位置，**反池化 max unpooling**再把值放回这些位置，其余位置补 0，这可以恢复空间尺寸，但不能恢复池化时丢掉的所有数值

![[CV导论/imgs/img7/image-17.png]]

## Transposed Convolution

![[CV导论/imgs/img7/image-11.png]]

对于输入的每一个位置的值，乘上卷积核里的权重，放到一个新的图里
重叠的部分就把得到的结果加起来，**转置卷积 transposed convolution**具有可学习权重，可以扩大空间尺寸，但通常不是原卷积的逆运算

之所以叫 transposed，考虑1维情况
![[CV导论/imgs/img7/image-12.png]]

假设输入为 $\vec{a}$，卷积核为 $\vec{x}$ ，对比正常的卷积，写为 Toeplitz 矩阵形式

![[CV导论/imgs/img7/image-13.png]]

于是可以构建这样的结构
红色为步长2的卷积，蓝色为步长2的反卷积，灰色为步长1的不变大小的卷积

![[CV导论/imgs/img7/image-14.png]]

相比于单纯的步长1的卷积，可以降低内存开销
通过 Downsampling 可以更快提高感受野，获得更好的 global context

为了信息压缩后能正常恢复，bottleneck 需要记住以下信息

- 全图范围内的特征信息 Global context
- 每个像素的空间信息，尤其是边缘 boundary 位置的
如果单纯的先压缩再恢复，那比较困难，所以需要一些结构上的优化

## UNet

![[CV导论/imgs/img7/image-15.png]]

在 encoder 和 decoder 中间加一些 skip link
把 encoder 中较高分辨率的特征通过 skip link 传给对应的 decoder，补充边缘和位置细节，让 bottleneck 不必独自保留全部空间信息，可以更多地提取全局语义

## Evaluation

如果用每个像素的准确性去评估图片分割效果，即每个像素都视作一个分类问题
图片中不同类别的事物的占比可能很不均匀
如果模型对于多数部分预测准确，但是少数部分不准确，那计算的 acc 也较高
这样总体像素准确率可能掩盖小类别上的错误，因此需要分别衡量各个类别的预测区域与真实区域是否重合

**Intersection over Union**

$$
\operatorname{IoU}=\frac{|\text{target}\cap\text{prediction}|}{|\text{target}\cup\text{prediction}|}
$$

![[CV导论/imgs/img7/image-16.png]]

对每个类别，把真实属于该类的像素与预测属于该类的像素视为两个集合，交集是共同覆盖的部分，并集是两者覆盖的全部区域，IoU 越大，重合越好

分别计算各类别的 IoU，再对类别取平均，得到 **mean IoU (mIoU)**，它不是简单平均每个实例的准确率，具体按单图还是整个数据集累计交并集，要遵循评测约定，数值高低也需要结合数据集判断

**Soft IoU Loss**

$$
IoU = \frac{I(X)}{U(X)}
$$

为了让区域重合程度也能参与训练，可以把硬分类 mask 换成预测概率，下面 $X_v$ 是像素 $v$ 的预测概率，$Y_v$ 是对应的 0/1 标签，用乘积和求和近似交集与并集

$$
I(X) = \sum_{v \in V} X_v * Y_v
$$

$$
U(X) = \sum_{v \in V} (X_v + Y_v - X_v * Y_v)
$$

将这个可微的重合率转换为“越小越好”的损失，常用 $L_{IoU}=1-\operatorname{IoU}$，实现时还需要加入小量或约定，处理分母为 0 的情况

$$
L_{IoU} = 1 - IoU = 1 - \frac{I(X)}{U(X)}
$$
