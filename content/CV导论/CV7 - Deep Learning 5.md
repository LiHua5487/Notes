---
date: 2025-04-09
tags:
  - CV
---
# Image Net Classification
## VGG Net

![[CV导论/imgs/img7/image.png|330x383]]

Difference to Alex Net
- deeper network
- smaller conv

## Receptive Field

![[CV导论/imgs/img7/image-1.png]]

假设卷积核大小为 5×5
卷积后的每一个 cell 都由前面的 5×5 的 cell 影响，即对应的感受野

如果一直盯着小的局部看，那得到的信息量很有限，判断效果差
所以最终的感受野要尽量扩大到全图
每次卷积和池化都会让感受野增长，最终看到比较完整的特征
而且池化对感受野的增长作用会更大

![[CV导论/imgs/img7/image-2.png]]

假设做了3次 3×3 卷积
那 A3 在 A2，A1, Input 上的感受野分别为  3×3，5×5，7×7
如果直接在 Input 使用 7×7 的卷积核，那感受野也是 7×7
但实际上效果并不一样，因为3次卷积会有3次 ReLU，能力更强
同时，3次 3×3 的参数个数比1次 7×7 更少，更好训练

1 × 1 的卷积和 FC 并不一样，因为 FC 的感受野是前面整个图


## Res Net

![[CV导论/imgs/img7/image-3.png|631x533]]

基本结构
- 一开始进行卷积和池化
- 而后堆叠 residual block
- 每堆叠几次用一次步长为 2，channel 数更高的卷积
	降低分辨率，但是提高了 channel 数，逐渐压缩提取特征
	总层数与每几次用一次步长2相关
- 最后再 Global Pooling 和 FC

为了提高 efficiency ，采用以下结构

![[CV导论/imgs/img7/image-4.png|424x306]]

先用 1×1 卷积降低 channel 数，最后再用 1×1 卷积升回去
牺牲了一部分感受野，但是有更多 ReLU ，更少的参数

假设输入大小为 $W\times W\times C$
这个结构的信息量类似于漏斗，中间3×3 卷积对应 bottle neck， 为 $W^2 \times \frac{C}{4}$

![[CV导论/imgs/img7/image-5.png]]

# Segmentation

分割图片，将图片分成不同区域，每个区域有一个语义 semantics

![[CV导论/imgs/img7/image-6.png]]

- Segmentation 划分区域
- Instance 同一区域区分不同个体
- Semantic 每个区域标出语义

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

- Informa on bottleneck
	the dimension of z space is much smaller than that of x 
- Get rid of redundant information via dimension reduction 
- The first step to all advanced segmentation on networks

如果输入中的信息量几乎没什么冗余，那直接压缩会造成信息丢失
但如果原来信息就存在大量冗余，那就可以压缩剔除
但是如果压缩太多，会出现不可逆转的信息丢失

比如对于一张试卷的照片，可以先提取出文字，再打印出来恢复


![[CV导论/imgs/img7/image-10.png]]

## Max Unpooling

反池化时，利用前面池化时传过来的位置信息，可以进行还原

![[CV导论/imgs/img7/image-17.png]]

## Transposed Convolution

![[CV导论/imgs/img7/image-11.png]]

对于输入的每一个位置的值，乘上卷积核里的权重，放到一个新的图里
重叠的部分就把得到的结果加起来

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
空间信息会通过 skip link 直接给到后边
这样，bottleneck 只需要记住 global context 就行

# Evaluate

如果用每个像素的准确性去评估图片分割效果，即每个像素都视作一个分类问题
图片中不同类别的事物的占比可能很不均匀
如果模型对于多数部分预测准确，但是少数部分不准确，那计算的 acc 也较高
但这是非常不准确的，所以需要统计每个事物类别的 acc

**Intersection over Union**
$$
IoU = \frac{target \cap prediction}{target \cup prediction}
$$

![[CV导论/imgs/img7/image-16.png]]

这样评估的单位就从像素变成了 mask
每个类别可能有很多 mask，计算这些 mask 的平均 acc
所有类别的 acc 算完了再整体做平均，得到 **mean IoU**

实际上 mean IoU 达到 50%+ 就很不错了

**Soft IoU Loss**
$$IoU = \frac{I(X)}{U(X)}$$where, $I(X)$ and $U(X)$ can be approximated as follows
$$I(X) = \sum_{v \in V} X_v * Y_v$$$$U(X) = \sum_{v \in V} (X_v + Y_v - X_v * Y_v)$$Therefore, the IoU loss $L_{IoU}$ can be defined as follows
$$L_{IoU} = 1 - IoU = 1 - \frac{I(X)}{U(X)}$$













































