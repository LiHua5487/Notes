---
date: 2025-03-05
tags:
  - CV
---
# Line Fitting

From edge detection we can get a series of points (0/1 mask).
Now for some cases, we wanna get a straight line.

## Least Square Method 

![](CV导论/imgs/img2/image-4.png)
We wanna minimize E to get a optimized result.
$residual\ 残差 \xrightarrow{L2\ norm\ 二范数}\ E (Energy)\ \xrightarrow{(m,b,E)}\ energy\ landscape\ \rightarrow\ global\ min$
![](img2/image-5.png)

P.S. more detailed calculation methods can be found in matrix cookbook.
This process is convex optimization. 凸优化（只有一个全局最小值）

- Limitations
	1. fails completely for vertical lines.
	2. not robust 鲁棒（抗干扰） when outliner 离群 exists

**For Problem1**, we can use $ax + by - d = 0$ to replace $y = mx + b$

![](img2/image-6.png)


$$
E = \sum(ax_i+by_i-d)^2 = 
||
\begin{bmatrix}
x_1 & y_1 & -1 \\
⋮ & ⋮ & ⋮ \\
x_n & y_n & -1
\end{bmatrix}

\cdot

\begin{bmatrix}
a \\
b \\
d
\end{bmatrix}
||^2
$$

记前后的矩阵分别为A和h，预期$E = Ah = 0$
约束：$a^2+b^2+d^2 = 1$，即 $||h|| =1$
find $(a, b, d)$ to minimize E, which is to minimize ||Ah||

case1 如果A是方阵
设A特征值，特征向量，有 $Ax = \lambda x$ 
则 $h = \sum\alpha_ix_i$
将A对角化，代入下式有
$$
||Ah||^2 = (A(\sum \alpha_i x_i))^2 = \sum(\alpha_i \lambda_i)^2
$$
$$
\because||h|| = 1\ \ \therefore\sum \alpha_i ^2 = 1
$$
取绝对值最小的特征的的系数为1，其余为0
h即为最小的特征值对应的特征向量

case 2如果A不是方阵
用singular value decomposition SVD 奇异值分解
$$A_{n×3} = U_{n×n}D_{n×3}V_{3×3}^T$$
同理，取h为 D中最小的奇异值的列，取V中的这一列
而一般奇异值会按从大到小排，取h为V中最后一列


**For Problem 2**

![](img2/image.png)

Robust to small noises but sensitive to outliers !
We need to replace L2 norm with a better method.
L1  is robust, but big mistake and small mistake are reflected as the same.

## RANSAC 随机抽样一致性

![](img2/image-7.png)

- Process
	1. 随便选2个点确定一个线，看这个线所确定的inlier占的比例
		![](img2/image-1.png)
	2. 先比inlier的比例，相同的话再用最小二乘/SVD比Energy
	3. 取最小的一组inlier点用最小二乘/SVD计算直线方程
	4. 用fit后的线再次确定inlier，重复上述流程，直到不变
		![](img2/image-2.png)

- hyper parameters 超参数
	1. number of hypothesis 假设（取样）的数目
	2. threshold 判断inlier的阈值，可以根据噪声的sigma确定

- P.S.
	inlier比例最大的不一定最后是最优的
	实现时可以不用循环取样，而是并行处理，利用Matrix或Tensor

- 至少需要多少取样？
	一次取n个点，取k次
	假设取一次inlier概率为 $w$
	一组点全部是inliner的概率 $P[all\ inliner] = w^n$ 
	如果一组点中有一个点是outliner, 那么我们称这组点fail. 
	k 组点全部fail的概率 $P[fail] = (1−w^n)^k$
	所以取n最小（即为2），k尽可能大

由上可得，最少需要的假设数 K 为
$$
K = \left\lceil \frac{\log(1 - p)}{\log\left(1 - w^n\right)} \right\rceil
$$
- p 为至少有一个取样是成功的概率
- w 为 inlier 比例
- n 为每次取点的个数

- Pros and Cons
	- Pros
		General method suited for a wide range of model fitting problems
		Easy to implement and easy to calculate its failure rate
	
	- Cons
		Only handles a moderate percentage of outliers without cost blowing up
		Many real problems have high rate of outliers (but sometimes selective choice of random subsets can help)

A voting strategy, **The Hough transform**, can handle high percentage of outliers

## Hough Transform

对于经过 $(x_1, y_1)$ 的所有直线，可以用 $y_1=mx_1+n$ 表示
将 m n 所有可能值列举出来，也就是将 m n 视为变量
那在 Hough 空间中就是一条直线，代表经过 $(x_1, y_1)$ 的直线簇
Hough 空间中不同直线的交点就代表同时经过 $(x_i, y_i)$ 的直线

![[CV导论/imgs/img2/image-21.png]]


Hough 空间中，统计每个位置被多少曲线覆盖的次数
次数越多，就代表这个点 $(m, n)$ 对应的直线拟合效果越好
对于噪声较大，即 outlier 较多的情况，也能应对

![[CV导论/imgs/img2/image-22.png]]

补充：
传统系统：modular-based system
模块接着模块，需要提高鲁棒性，不然一步错后边全完了
需要post processing 后处理
所以还需要 end-to-end system 端到端系统

# Corner Detection

In addition to edges, keypoints are also important to detect.

## Keypoint and Corner

What Points are Keypoints?
- Requirements
	1. Saliency 显著性
	2. Repeatability: detect the same point independently in both images
	3. Accurate localization
	4. Quantity: sufficient number

- For a keypoint detector to be repeatable, it has to be invariant 不变的 to
	Illumination 
	Image scale 
	Viewpoint

- Corners are such kind of keypoints, because they are 
	Salient
	Repeatable (one corner would still be a corner from another viewpoint)
	Sufficient (usually an image comes with a lot of corners
	Easy to localize.

The key property of a corner: In the region around a corner, image gradient has two or more dominant directions.

Accordingly, we can build **Harris Corner Detector**.

## Process

First, we get a slide window 滑动窗口（实际实现是并行的）
![](img2/image-3.png)

Suppose that we have the following image as input.

![|209x209](img2/image-10.png)

fix a slide window at $(x_0, y_0)$ , and move along a direction to $(x_0+u, y_0+u)$
![](img2/image-8.png)

![](img2/image-9.png)

Regard slide window as a function,
and define the square of intensity change as $D(x,y)$
![](img2/image-11.png)

More specifically
$$
\begin{align}
w'(x,y) = w(x-x_0,y-y_0) = w(x_0-x,y_0-y) \\
\sum w'(x,y)D(x,y) = \sum w(x_0-x,y_0-y)D(x,y) = w*D\ 卷积
\end{align}
$$

use First-order Taylor expansion

$$ 
I[x+u, y+v] - I[x, y] \approx I_xu + I_yv
$$

$$
\therefore \, D(x, y) = (I[x+u, y+v] - I[x, y])^2 \approx (I_xu + I_yv)^2 =
\begin{bmatrix}
u & v
\end{bmatrix}
\begin{bmatrix}
I_x^2 & I_xI_y \\
I_xI_y & I_y^2
\end{bmatrix}
\begin{bmatrix}
u \\
v
\end{bmatrix}
$$

$$
\therefore \, E_{(x_0, y_0)}(u, v) = w * D = 
\begin{bmatrix}
u & v
\end{bmatrix}
w \cdot
\begin{bmatrix}
I_x^2 & I_xI_y \\
I_xI_y & I_y^2
\end{bmatrix}
\begin{bmatrix}
u \\
v
\end{bmatrix}
$$

$$
Let\ M(x, y) = w \cdot
\begin{bmatrix}
I_x^2 & I_xI_y \\
I_xI_y & I_y^2
\end{bmatrix}
= 
\begin{bmatrix}
w \cdot I_x^2 & w \cdot (I_xI_y) \\
w \cdot (I_xI_y) & w \cdot I_y^2
\end{bmatrix}
$$

Note that $M(x, y)$ is a 2×2 symmetry matrix, suppose its eigenvalue is $\lambda_1$ , $\lambda_2$

$$
M(x, y) = w \cdot 
\begin{bmatrix}
I_x^2 & I_xI_y \\
I_xI_y & I_y^2
\end{bmatrix} = R^{-1}
\begin{bmatrix}
\lambda_1 & 0 \\
0 & \lambda_2
\end{bmatrix}
R
\quad (\lambda_1 \geq 0, \; \lambda_2 \geq 0)
$$

$$
\therefore \, E_{(x_0, y_0)}(u, v) \approx \lambda_1u_R^2 + \lambda_2v_R^2 \quad \text{where} \quad 
\begin{bmatrix}
u_R \\
v_R
\end{bmatrix}
= R
\begin{bmatrix}
u \\
v
\end{bmatrix}
$$
 $\lambda_1$ , $\lambda_2$ reflect the change along two directions
The energy landscape is a paraboloid like this

![|372x170](img2/image-12.png)

We can classify the region according to the value of  $\lambda_1$ and $\lambda_2$

![](img2/image-13.png)

Here is a simplified version to judge whether it's a corner

## Corner response function

![|430x292](img2/image-14.png)

setting coefficient $\alpha$ and threshold $t$ appropriately, 
when $\theta>0$ , the area is classified as a corner. 

Note that we can pick the function of slide window.

![](img2/image-15.png)

## Summary

Suppose that we have an input image.

![|203x154](img2/image-16.png)

- Process
	1. Image derivatives 
		![](img2/image-17.png)
	
	2. Square of derivatives 
		![](img2/image-18.png)
	
	3. Rectangle window or Gaussian filter 
		![](img2/image-19.png)
	
	4. Corner response function
		$\theta = g(I_x^2)g(I_y^2) - [g(I_xI_y)]^2 - \alpha[g(I_x^2) + g(I_y^2)]^2 - t$
	
	5. Thresholding to obtain a binary mask
		$\theta(x0,y0) > 0$
	
	6. Non-maximum suppression

We can get the output

![|243x190](img2/image-20.png)




