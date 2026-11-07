---
date: 2025-02-26
tags:
  - CV
---

# Image as Function

一张图像可以看作一个函数，输入像素的位置 $(x,y)$，输出这个位置的颜色或亮度，也可以直接把它存成一个数组

- **灰度图 grayscale image**：每个像素用一个数表示亮度，称为灰度值 intensity，通常取 $0\sim255$，也可以缩放到 $0\sim1$，数值越大越亮
- **RGB 图像**：每个像素分别记录红、绿、蓝三个数，因此有 3 个通道 channel
- **分辨率 resolution**：图像在宽、高方向上有多少个像素

如果图像高为 $H$、宽为 $W$，灰度图可以存成 $H\times W$ 的矩阵，RGB 图像可以存成 $H\times W\times3$ 的张量 tensor，这里可以先把张量理解为多维数组

## Gradient

如果某处两侧的亮度差很多，就可能是一条边缘，那怎么衡量“变化有多大”？可以求图像在水平、竖直两个方向的导数，分别记为 $I_x$、$I_y$，把它们放在一起就得到**梯度 gradient**

$$
\begin{aligned}
\nabla I &= \begin{bmatrix}I_x & I_y\end{bmatrix}^T \\
\|\nabla I\| &= \sqrt{I_x^2+I_y^2} \\
\theta &= \operatorname{atan2}(I_y,I_x)
\end{aligned}
$$

梯度的方向是亮度增加最快的方向，模长表示变化有多剧烈，通常与边缘方向垂直

![[download.jpg|332x62]]

![[download-1.jpg|487x128]]

![[download-2.jpg|464x52]]

真实图像由离散像素组成，不能直接对连续函数求导，可以通过相邻像素的差值近似导数，下面的例子展示了图像和对应的梯度

![[download-3.jpg|544x186]]

![[download-4.jpg|478x294]]

![[CV导论/imgs/img/image.png]]

## Filter

**滤波 filter**：根据原图中一个局部区域的像素值，计算新图中对应位置的像素值，既可以提取边缘、角点等信息，也可以用于去噪、锐化等处理

## Convolution

### 1D Discrete-Space Systems

先考虑一维信号，比如一串随时间变化的亮度值，如果想让它更平滑，可以把每个位置及其左右邻居的值取平均

$$
h[n]=0.2\big(f[n-2]+f[n-1]+f[n]+f[n+1]+f[n+2]\big)
$$

这里 $f$ 是输入、$h$ 是输出，相当于拿权重为 $[0.2,0.2,0.2,0.2,0.2]$ 的窗口在输入上滑动，每次把对应位置相乘后求和，这组权重就称为**卷积核 kernel**

![[CV导论/imgs/img/image-1.png|291x113]]

![[CV导论/imgs/img/image-2.png]]

比如输入为 `[1,2,3,4,5]`，卷积核为 `[1,0,-1]`，每次向右移动 1 格，这个移动距离称为**步长 stride**，不在两端补值时，三个输出分别为

$$
\begin{aligned}
h_1 &= 1\times1+2\times0+3\times(-1)=-2 \\
h_2 &= 2\times1+3\times0+4\times(-1)=-2 \\
h_3 &= 3\times1+4\times0+5\times(-1)=-2
\end{aligned}
$$

于是得到 `[-2,-2,-2]`，这个核实际上在比较窗口两侧的差异

>严格来说，上面不翻转卷积核的滑动计算叫**互相关 cross-correlation**，数学上的卷积要先把核翻转，深度学习中通常沿用“卷积”的叫法，权重由训练学习得到时，两种约定可以表达同一类运算，但对于一个给定的核，计算结果一般不同

### Convolution Theorem

一个信号既可以从“各个位置的取值”来描述，也可以从“包含哪些频率”来描述，傅里叶变换 $\mathcal F$ 就用于在这两种表示之间转换，变化缓慢的部分对应低频，变化快的细节和部分噪声对应高频

**卷积定理**：空间域或时域中的卷积，对应频域中的逐点相乘

$$
\begin{aligned}
\mathcal F(f*g)&=\mathcal F(f)\mathcal F(g) \\
f*g&=\mathcal F^{-1}\big(\mathcal F(f)\mathcal F(g)\big)
\end{aligned}
$$

所以可以先把输入和卷积核变换到频域，逐点相乘后再逆变换回来，这为大尺寸卷积提供了一种高效计算方式，实际是否更快还与核大小、输入大小和实现有关，离散计算时还需要适当补零，避免首尾绕回造成循环卷积

比如低通滤波会削弱高频成分，保留变化缓慢的部分，因此图像会变平滑

![[CV导论/imgs/img/1.png|445x282]]

下面是一些常见滤波核及其效果

![[CV导论/imgs/img/2.png]]

### 2D Discrete-Space Systems

二维情况也是一样，只不过窗口从一段数字变成一块矩形区域，拿卷积核在图像上滑动，每次把对应像素乘以权重再求和，得到一个输出像素

![[CV导论/imgs/img/3.png|448x268]]

![[CV导论/imgs/img/image-3.png|484x226]]

![[CV导论/imgs/img/image-4.png|487x227]]

## Binarization

**二值化 binarization**：设置阈值 $\tau$，把每个位置的值变成 0 或 1，比如取 $\tau=100$

$$
h[m,n]=\begin{cases}
1,&f[m,n]>\tau \\
0,&f[m,n]\leq\tau
\end{cases}
$$

这样就能用一张只含 0、1 的图，标记哪些位置满足条件，这种图称为二值掩码 binary mask，二值化本身是非线性操作

比如先通过滤波计算变化程度，再用阈值筛出变化明显的位置，就可以初步检测边缘

![[CV导论/imgs/img/image-6.png|221x113]] ![[CV导论/imgs/img/image-7.png|114x113]]

# Edge Detection

## Definition

**边缘 edge**：图像中沿某个方向亮度变化明显、沿其垂直方向变化较小的区域，可能来自物体的遮挡边界、表面朝向变化、颜色变化或光照变化，因此检测出的边缘不一定都是物体的轮廓

研究一个检测任务时，可以先明确问题和定义，再决定评价标准，最后设计具体算法

## Evaluation Metrics

先把预测结果与真实标注 ground truth 比较，P/N 表示预测为正例/负例，T/F 表示预测正确/错误

- **TP**：确实是边缘，也预测为边缘
- **FP**：不是边缘，却预测为边缘，即误检
- **FN**：确实是边缘，却没有检测出来，即漏检
- **TN**：不是边缘，也没有预测为边缘

![[CV导论/imgs/img/image-8.png]]

两个常见指标是**精确率 precision**和**召回率 recall**

$$
\begin{aligned}
\operatorname{Precision}&=\frac{TP}{TP+FP} \\
\operatorname{Recall}&=\frac{TP}{TP+FN}
\end{aligned}
$$

- Precision：预测出的边缘里，有多少是真的，误检越少越好
- Recall：实际存在的边缘里，有多少被找到了，漏检越少越好

改变检测阈值会得到不同的 precision、recall，可以绘制 PR 曲线来观察两者的关系，除此之外，还需要关注

- **定位精度 good localization**：检测位置要尽量贴近真实边缘
- **单一响应 single response constraint**：同一条真实边缘不要被重复检测成多条线

## Smoothing and Gradient

理想的边缘表现为亮度的突然变化

![[download-3-1.jpg|461x158]]

但噪声也会导致相邻像素变化剧烈，直接求梯度会把噪声也检测出来

![[CV导论/imgs/img/image-9.png]]

所以先用**高斯滤波 Gaussian filter**平滑图像，它是一种低通滤波，离中心越近的像素权重越大

![[CV导论/imgs/img/image-10.png]]

![[CV导论/imgs/img/image-11.png|424x311]]

如果先平滑再求导，需要两步计算，不过根据卷积的求导性质

$$
\frac{d}{dx}(f*g)=f*\frac{dg}{dx}
$$

可以直接用高斯核的导数与原图卷积，把这两步合在一起

![[CV导论/imgs/img/image-12.png|390x237]]

![[CV导论/imgs/img/image-13.png|450x354]]

此时已经能看到边缘，但一条边缘可能有好几个像素宽，还需要细化和连接

## Algorithm

### Non-Maximum Suppression (NMS)

**非极大值抑制**：沿梯度方向比较当前像素与两侧的梯度幅值，只保留局部最大的像素，把其余位置置零，这样可以将较宽的响应细化成线

![[CV导论/imgs/img/image-14.png]]

梯度方向不一定正好经过相邻像素的中心，如果比较位置落在网格之间，可以用附近像素进行**双线性插值 bilinear interpolation**：先沿一个方向做线性插值，再沿另一个方向插值，估算这个位置的梯度幅值

![[CV导论/imgs/img/image-15.png]]

先水平后竖直、先竖直后水平，得到的结果相同，也可以把梯度方向近似到水平、竖直和两个对角线方向，再与对应的两个邻居比较，省掉插值

![[CV导论/imgs/img/image-16.png]]

这里比较的是梯度方向两侧的响应，若直接要求比周围 8 个像素都大，会变成更严格的二维局部极大值筛选，可能把连续边缘上的点也去掉

![[CV导论/imgs/img/image-17.png|357x186]]

### Hysteresis Thresholding

细化后，强边缘比较可信，但一些较弱的真实边缘也需要保留，可以使用**滞回阈值 hysteresis thresholding**，设置高、低两个阈值 $T_H>T_L$

- 梯度幅值高于 $T_H$：作为强边缘保留
- 低于 $T_L$：直接去掉
- 在两者之间：只有能通过相邻候选边缘连接到强边缘时才保留，通常按 8 邻域搜索

相当于先找到可信的起点，再把与其相连的弱边缘接起来，而孤立的弱响应更可能是噪声，阈值需要根据图像和梯度的数值范围选择

![[CV导论/imgs/img/image-18.png|307x178]]

图中的 A 能接到强边缘，因此保留；B 不能接到，则去掉，最终得到比较细且连续的边缘

![[CV导论/imgs/img/image-19.png|339x80]]

将高斯平滑、梯度计算、非极大值抑制和双阈值连接组合起来，就是 **Canny Edge Detector**

![[CV导论/imgs/img/image-20.png|305x167]]

## Summary

| 处理步骤 | 作用 |
| --- | --- |
| 灰度化 | 将 RGB 图像转换为亮度表示，便于进行后续计算 |
| 高斯平滑 | 减少噪声对梯度的干扰 |
| 计算梯度 | 得到 $G_x$、$G_y$，进而计算梯度幅值和方向 |
| 非极大值抑制 | 沿梯度方向保留局部最大响应，细化边缘 |
| 滞回阈值 | 从强边缘出发连接弱边缘，去掉孤立响应 |
