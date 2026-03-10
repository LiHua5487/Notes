
# Panorama

想要根据不同角度拍摄的照片拼接 stitch 成一个全景照片 panorama ，需要进行以下步骤
- Extract feature points：找到图片特征点，这可以使用 SIFT 算子等
- Feature matching：将两个图片中的特征点对应起来
- Solve transformations：根据对应点对图像进行变换拼接
- Blend Images：进一步渲染以使过渡平滑

## Solve transformations

回顾一下两种基本的变换
- Image Filtering：改变图像的像素值，即值域 Range
- Image Warping：改变图像的形状，即定义域 Domain

![[CV/img/img4/image.png]]

在拼接时，我们先只考虑变形，有很多种变形方式，将其表述为一个变换 $T$ 

![[CV/img/img4/image-1.png]]

假设只进行仿射变换，即只有平移和旋转，那就可以表述为 $p'=Ap+b$ ，在齐次坐标下对应的变换就是

$$\begin{bmatrix}
x' \\
y' \\
1
\end{bmatrix}
=
\begin{bmatrix}
a & b & c \\
d & e & f \\
0 & 0 & 1
\end{bmatrix}
\begin{bmatrix}
x \\
y \\
1
\end{bmatrix}
= T
\begin{bmatrix}
x \\
y \\
1
\end{bmatrix}$$

一些常见的仿射变换如下

![[CV/img/img4/image-2.png]]

但一般情况下，T 有 6 个自由度，假设找到一组对应点 $[x_i', y_i'] \leftrightarrow [x_i, y_i]$ ，则可列出以下方程

$$\begin{bmatrix}
x_i' \\
y_i'
\end{bmatrix} = 
\begin{bmatrix}
a & b \\
d & e
\end{bmatrix}
\begin{bmatrix}
x_i \\
y_i
\end{bmatrix} +
\begin{bmatrix}
c \\
f
\end{bmatrix}$$

可见每组对应点能提供 2 个方程，所以至少需要 3 组对应点，把这些方程都转换成矩阵形式，就可得

$$
\begin{align*}
&\begin{bmatrix}
\vdots \\
x_i' \\
y_i' \\
\vdots
\end{bmatrix} =
&&\begin{bmatrix}
& & \vdots & &\\
x_i & y_i & 0 & 0 & 1 & 0 \\
0 & 0 & x_i & y_i & 0 & 1 \\
& &\vdots & &
\end{bmatrix}
&&\begin{bmatrix}
a \\
b \\
d \\
e \\
c \\
f
\end{bmatrix} \\
&\mathbf{b}_{2n \times 1} = &&\mathbf{A}_{2n \times 6} \quad &&\mathbf{t}_{6 \times 1}
\end{align*}
$$

实际求解中，可以利用最小二乘估计，即最小化下式

$$E = \| \mathbf{A} \mathbf{t} - \mathbf{b} \|_2^2 = \mathbf{t}^T \mathbf{A}^T \mathbf{A} \mathbf{t} - 2 \mathbf{t}^T\mathbf{A}^T \mathbf{b} + \mathbf{b}^T \mathbf{b}$$

对其求导可得极值点，即解以下方程

$$\mathbf{A}^T \mathbf{A} \mathbf{t} = \mathbf{A}^T \mathbf{b}$$

---

上述方法只是针对于仿射变换，但还可能需要透视变换（一种特殊的投影变换，对应透视的投影方式）等

![[CV/img/img4/image-3.png]]

>单应性 Homography：描述两个平面之间的投影变换的数学关系

对于一般情况需要把 T 中的最后一行也换成未知量

$$\begin{bmatrix}
x_i' \\
y_i' \\
1
\end{bmatrix} =
\begin{bmatrix}
h_{00} & h_{01} & h_{02} \\
h_{10} & h_{11} & h_{12} \\
h_{20} & h_{21} & h_{22}
\end{bmatrix}
\begin{bmatrix}
x_i \\
y_i \\
1
\end{bmatrix}$$

由于用的是齐次坐标， $[x\ y\ 1]$ 和 $[wx\ wy\ w]$ 实际上是一样的，所以实际上 T 只有 8 个自由度，至少需要 4 组点，为此可以规定 $h_{22}$ 为 1 ，也可以规定所有元素平方和为 1，一般采用后者，即

$$||h||^2=1,\quad \text{where } h=[h_{00} \cdots h_{22}]$$

把方程表达为矩阵形式得

$$\begin{bmatrix}
x_1 & y_1 & 1 & 0 & 0 & 0 & -x_1'x_1 & -x_1'y_1 & -x_1' \\
0 & 0 & 0 & x_1 & y_1 & 1 & -y_1'x_1 & -y_1'y_1 & -y_1' \\
& & & & &\vdots & & &\\
x_n & y_n & 1 & 0 & 0 & 0 & -x_n'x_n & -x_n'y_n & -x_n' \\
0 & 0 & 0 & x_n & y_n & 1 & -y_n'x_n & -y_n'y_n & -y_n'
\end{bmatrix}
\begin{bmatrix}
h_{00} \\
h_{01} \\
h_{02} \\
h_{10} \\
h_{11} \\
h_{12} \\
h_{20} \\
h_{21} \\
h_{22}
\end{bmatrix} =
\begin{bmatrix}
0 \\
0 \\
\vdots \\
0
\end{bmatrix}$$

可总结为以下方程

$$\mathbf{A}_{2n \times 9} \mathbf{h}_9 = \mathbf{0}_{2n}, \quad||h||^2=1$$

采用拉格朗日乘数法

$$E = \| \mathbf{A} \mathbf{h} \|^2 + \lambda (\| \mathbf{h} \|^2 - 1) = \mathbf{h}^T \mathbf{A}^T \mathbf{A} \mathbf{h} + \lambda \mathbf{h}^T \mathbf{h} - \lambda$$

即需求解以下方程

$$\mathbf{A}^T \mathbf{A} \mathbf{h} = \lambda \mathbf{h}$$

---

值得注意的是，为了更好的拼接出全景图，一般要求相机拍摄时只进行旋转，尽量不进行平移：由于场景深度，平移相机会带来视差，比如一个图里 A 比 B 大，另一个图里 B 比 A 大，但是拼接时是对整个图片进行变换，很难同时保证 A 和 B 以及背景部分都能完全对应上，这也是为什么要使用最小二乘估计而非直接求解

如果拍摄对象为一个平面，不存在深度信息，那相机平移就没事了

## Feature Matching

在找对应点时，可以按照最近邻的方式查找，即对于一个图片的点，找到另一个图片中 L2 distance 最小的点

但这可能存在一些 outlier ，如果 outlier 参与了变换矩阵的计算，结果会有偏差，可以使用 RANSAC 进行剔除，先随机选 4 组点计算一个矩阵，而后计算 inlier 比例，选择 inlier 比例最大的情况，用这些 inlier 计算矩阵

## Image Blending

之前的拉普拉斯金字塔渲染存在一个问题，边缘虽然平滑了，但是两个图片内部的色调还是不一致，为此可以使用 **Poisson Image Editing**

![[CV/img/img4/image-4.png]]

假设我们要从源图像 $g$ 上扣出一个区域 $\Omega$ 贴到目标图像 $f$ 上，渲染得到输出图片 $f^*$ （只需关注 $\Omega$ 部分长啥样，剩余部分拷贝目标图像即可）

我们希望保留 $\Omega$ 部分的纹理等信息，这可以用梯度来代表，问题可以变为最小化梯度差异

$$E = \min_f \sum_{(i,j) \in \Omega} \|\nabla f(i,j) - \nabla g(i,j)\|^2$$

同时还要保证过渡平滑，可以规定为 $\Omega$ 部分的边缘处 $\partial \Omega$ 要和目标图像一样

$$\forall (i,j)\in \partial \Omega,\quad f(i,j)=f^*(i,j)$$

在求解时，可以采用最小二乘法，把二者合并到一个式子里，最小化以下目标

$$\min_f \|A f - G\|^2 + \lambda \|B f - F\|^2$$

- $\|A f - G\|^2$ 代表梯度约束，$A$ 是求梯度的矩阵形式，$G$ 是源图像的梯度矩阵
- $\|B f - F\|^2$ 代表边界约束，$B$ 是边界选择矩阵，$F$ 是目标图像边界的像素值矩阵

可见 $A$ 和 $B$ 都是非常稀疏的，具体计算参考 Large-scale Sparse Matrix 的处理方式

---

在将 $\nabla f$ 离散化时（以 $x$ 方向为例），一个简单的想法是
$$\frac{\partial f}{\partial x} \approx f(x + 1,y) - f(x,y)$$
但下面的方法更逼近，因为泰勒展开后可以消掉二阶导那一项
$$\frac{\partial f}{\partial x} \approx \frac{f(x + 1,y) - f(x-1,y)}{2}$$
而拉普拉斯算子 $\nabla^2$ 的离散化是
$$\nabla^2 f \approx f(x + 1,y) + f(x - 1,y) + f(x,y + 1) + f(x,y - 1) - 4f(x,y)$$





