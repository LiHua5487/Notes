---
date: 2025-02-26
tags:
  - CV
---
# Image as Function

- Map the image $(x,y)$ position to a set of values
## Basic Words

-  **Intensity** 灰度 (1channel )
	 ![[Pasted image 20250228094402.jpg|242x78]]
- **RGB** (3 channel)
	![[Pasted image 20250228094424.jpg|175x109]]
- **Resolution** 分辨率 
- **Matrix** 矩阵 $(H \cdot W \cdot 1)$
- **Tensor** 张量 $(H \cdot W \cdot 3)$

## Gradient 梯度

- Mathematical formulations
	![[download.jpg|332x62]]
	![[download-1.jpg|487x128]]
	![[download-2.jpg|464x52]]

Gradient shows the edges of an image.
![[download-3.jpg|544x186]]

- **Example**
	![[download-4.jpg|478x294]]

P.S.
![[CV导论/imgs/img/image.png]]

## Filter 滤波器


- Definition
	Form a new image whose pixels are a combination of original pixel values
- Goals
	- Extract useful info (edges, corners ... )
	- Modify or enhance image properties (super-resolution, de-noising ... )

## Convolution 卷积

### 1D Discrete-Space Systems

- Mathematical formulations
	![[CV导论/imgs/img/image-1.png|291x113]]

Let $h[n] = 0.2 \cdot (f[n-2]+f[n-1]+f[n]+f[n+1]+f[n+2])$
We can get a **Linear Filter** 线性滤波器

![[CV导论/imgs/img/image-2.png]]
This process is called **Convolution**.

- Example
	- 基本数据
		- 输入数据（input）：`[1, 2, 3, 4, 5]`
		- 卷积核（kernel）：`[1, 0, -1]`
		- 滑动步长（stride）：1（每次滑动 1 步）
		我们的任务是用卷积核“滑过”输入数据，滑一次就算出一个结果。
	- 滑动操作
		1. **第一步：把卷积核对齐输入的开头部分** 卷积核 `[1, 0, -1]` 和输入数据的前 3 个数字 `[1, 2, 3]` 对齐：
		    `input:  [1,   2,   3,  4,  5]`
		    `kernel: [1,   0,  -1]`
		    然后按位相乘并累加：
		    `（1×1） + （2×0） + （3×-1） = 1 + 0 - 3 = -2`
		    所以第一个结果是 `-2`。
		2. **第二步：滑动卷积核，向右移动 1 步** 卷积核现在和输入数据的 `[2, 3, 4]` 对齐：
		    `input:  [1,   2,   3,   4,  5]`
		    `kernel:      [1,   0,  -1]`
		    再次按位相乘并累加：
		    `（2×1） + （3×0） + （4×-1） = 2 + 0 - 4 = -2`
		    第二个结果还是 `-2`。
		3. **第三步：继续滑动卷积核，向右再移动 1 步** 卷积核现在和输入数据的 `[3, 4, 5]` 对齐：
		    `input:  [1,   2,   3,   4,  5]`
		    `kernel:          [1,   0,  -1]`
		    再按位相乘并累加：
		    `（3×1） + （4×0） + （5×-1） = 3 + 0 - 5 = -2`
		    第三个结果还是 `-2`。
	- 最终结果
		把每一次滑动计算的结果放在一起，我们得到输出结果：
		`output = [-2, -2, -2]`

The weight of convolution kernel can be modified by Learning.

#### Convolution Theorem 卷积定理
$$
\begin{align}
\mathcal{F}(f*g) = \mathcal{F}(f)\mathcal{F}(g) \\ 
\therefore h = \mathcal{F}^{-1}(\mathcal{F}(f)\mathcal{F}(g))
\end{align}
$$
即 $时域的卷积 \rightarrow 频域的乘积 \xrightarrow{求逆} 卷积的结果$
其中傅里叶变换 $\mathcal{F}(f)$ : $时域 \xrightarrow{\mathcal{F}} 频域$

卷积与傅里叶变换的关系
根据卷积定理，卷积在时域中是傅里叶变换的乘法，即：
$$
f(x) * g(x) \xrightarrow{\text{Fourier Transform}} F(u) \cdot G(u)
$$
这里：
- $f(x) * g(x)$ 是信号 $f(x)$ 和滤波器 $g(x)$ 的卷积。
- $F(u) \cdot G(u)$ 是它们在频率域中的对应元素逐点相乘。

- 具体作用：
	- 傅里叶变换可以将卷积操作从空间域（复杂的滑动窗口）转换为频域的点乘（高效）。
	- CNN 的卷积操作本质上等价于在图像或信号的空间域执行固定大小的滤波。而通过傅里叶变换，这种卷积可以通过频域乘法计算并快速实现。
	- 高效实现卷积运算

卷积神经网络中，卷积操作是最核心的部分，但对大图像的卷积会非常耗时
通过傅里叶变换和快速傅里叶变换（FFT）可以加速这一过程

- 计算过程：
	1. 将输入图像和卷积核转换到频域中（使用傅里叶变换）
	2. 在频域中执行元素逐点相乘
	3. 使用逆傅里叶变换将结果返回到空间域中
- Use Case:
	在大型卷积操作中，频域方法比直接运算更快，特别是当卷积核较大时

公式加速的视图：
$$
\text{conv2d}(I, K) \approx \mathcal{F}^{-1}\left(\mathcal{F}(I) \cdot \mathcal{F}(K)\right)
$$

- Example 
	低通滤波：保留低频，消去高频
	![[CV导论/imgs/img/1.png|445x282]]
	根据卷积定理 $\mathcal{F}(f*g) = \mathcal{F}(f)\mathcal{F}(g)$
	这意味着 $\mathcal{F}(f)$ 中的高频部分在 $\mathcal{F}(f*g)$ 中变为0

Basic Filter Types
![[CV导论/imgs/img/2.png]]

### 2D Discrete-Space Systems

- Mathematical formulations
	![[CV导论/imgs/img/3.png|448x268]]

- Example
	 将卷积核在输入图片上滑动
	![[CV导论/imgs/img/image-3.png|484x226]]
	
	最终得到结果
	![[CV导论/imgs/img/image-4.png|487x227]]

## Binarization 二值化

Define a threshold $\tau$, e.g., $\tau = 100$.
$$
h[m, n] =
\begin{cases} 
1, & f[n, m] > \tau, \\
0, & \text{otherwise.}
\end{cases}
$$
P.S. This process is not a linear system.

- Example
	Process 
		Original $\rightarrow$ Convolution 
		Original $\rightarrow$ Binarization
	Result
		![[CV导论/imgs/img/image-6.png|221x113]]![[CV导论/imgs/img/image-7.png|114x113]]
	We can use the above process to conduct Edge Detection.

# Edge Detection

- A common scientific study process
	Problem Formulation $\rightarrow$ Definition $\rightarrow$ Evaluation Metrics $\rightarrow$ Algorithm

## Definition

An edge is defined as a region in the image where there is a “significant” change in the pixel intensity values (or having high contrast) along one direction in the image, and almost no changes in the pixel intensity values (or low contrast) along its orthogonal direction.

## Evaluation Metrics

![[CV导论/imgs/img/image-8.png]]

- T/F 实际是不是边缘
- P/N 模型预测是不是边缘

- **Metrics** 指标
	- **Precision** 精确率 
		衡量  “模型预测对了的边缘占预测出的边缘的比例”：
		$$ Precision = \frac{TP}{TP + FP} $$

	 - **Good Localization**
		衡量 “检测到的边缘的位置准确性“
		$$ Recall = \frac{TP}{TP + FN} $$
		- 减少检测边缘和真实边缘之间的位置误差，使边缘的定位更准确
		- 高 Recall 的意义：模型能够检测到尽量多的真实边缘（漏报少）
		- 但高 Recall 不一定说明模型的预测质量高，可能同时带来了很多误报

	-  **Single Response Constraint** 单一响应约束
		在边缘检测问题中，一个真实边缘区域可能会被模型检测到多次（即有多个检测预测重叠），这会显得冗余，甚至降低低性能指标。
		约束模型对于同一个边缘区域只响应一次，防止反复检测同一处边缘。
	Further Metrics such as **PR Curve** could be used.

What Causes An Edge?
	Depth/ Surface orientation/ Surface color/ Illumination discontinuity

## Brief review 

- Characterizing Edges
	![[download-3-1.jpg|461x158]]
- Effects of Noises
	![[CV导论/imgs/img/image-9.png]]
- A classic low-pass filter - **Gaussian Filter**
	![[CV导论/imgs/img/image-10.png]]
- Smoothing by a Low-Pass Filter
	![[CV导论/imgs/img/image-11.png|424x311]]
	Seems that we can save one operation!
- **Derivative Theorem of Convolution**
$$
\frac{d}{dx}(f * g) = f * \frac{d}{dx}g
$$
	 ![[image-12.png|390x237]]


- Above all, we can get this :
	![[CV导论/imgs/img/image-13.png|450x354]]
	But there is still some edges that is wider than 1 pixel.

## Algorithm

#### Non-Maximal Suppression (NMS) 非最大化抑制

To get a 1 pixel-wide line, we only save the point with maximum gradient.

![[CV导论/imgs/img/image-14.png]]

#### Bilinear Interpolation 双线性插值

![[CV导论/imgs/img/image-15.png]]
P.S. Also, we can do this by using $Q_{11}Q_{12}$ and $Q_{21}Q_{22}$, but the results are the same.

#### A Simplified Version of NMS

- We can simply compare a pixel with its surrounding 8 pixels.

![[CV导论/imgs/img/image-16.png]]

Then we can save the process of Bilinear Interpolation.

- After NMS, we get this !
	![[CV导论/imgs/img/image-17.png|357x186]]
	Now we need to link the edges.

#### Hysteresis Thresholding 滞回阈值

- Use a high threshold ($maxVal$) to start edge curves and a low threshold ($minVal$) to continue them.
	- Pixels with gradient magnitudes > $maxVal$ should be reserved.
	- Pixels with gradient magnitudes < $minVal$ should be removed. 
- How to decide $maxVal$ and $minVal$? Examples: 
	- $maxVal$ = 0.3 × average magnitude of the pixels that pass NMS. 
	- $minVal$ = 0.1 × average magnitude of the pixels that pass NMS.
- Now using the direction information and the lower threshold, we'll "grow" these edges.
	- If the current pixel is not an edge, check the next one. 
	- If it is an edge, check the two pixels in the direction of the edge (i.e., perpendicular to the gradient direction). 
	- If either of them (or both) passes the threshold, mark them as edges.

- Example
	![[CV导论/imgs/img/image-18.png|307x178]]
	A is ok, B is not ok.

- At last we can get an edge like this :
	![[CV导论/imgs/img/image-19.png|339x80]]
	This is called **Canny Edge Detector**.

- Eventually, the picture is processed like this :
	![[CV导论/imgs/img/image-20.png|305x167]]

# Conclusion

**Edge Detection Process**

| 处理步骤                  | 是否涉及卷积?                      | 作用                                                                                                                       |
| --------------------- | ---------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| 灰度化                   | ❌ 不需要卷积                      | 仅需将RGB图片转为灰度图，像素值降维                                                                                                      |
| 去噪（高斯滤波）              | ✅ 需要卷积                       | 平滑图像，去除高频噪声，避免噪声干扰边缘检测                                                                                                   |
| 梯度计算（Sobel/Laplacian） | ✅ 需要卷积                       | 分别计算水平方向 ![$G_x$](https://latex.codecogs.com/svg.image?G_x) 和垂直方向 ![$G_y$](https://latex.codecogs.com/svg.image?G_y) 的梯度 |
| 计算梯度幅值和方向             | ❌ 基于高斯滤波和Sobel算子结果，后续计算为代数运算 | 根据卷积的结果计算边缘强度（幅值）和方向（方向角度）                                                                                               |
| 非极大值抑制（NMS）           | ❌ 运用卷积结果                     | 对卷积后的梯度幅值进行抑制处理，细化边缘。                                                                                                    |
| 滞回阈值（Hysteresis）      | ❌ 运用卷积结果                     | 使用两个预定阈值连接边缘，提取更正确的边缘区域                                                                                                  |
