
# Camera Calibration

## 线性标定

相机标定：已知 n 个点的 3 维世界坐标 $X_i$ ，以及投影到图像上的坐标 $x_i$ ，要计算出相机的内参和外参

![[CV/img/img5/image.png]]

>世界坐标 $X_i$ 的获取：可以在场景中放置一系列不平行的平板（称为标定板 calibration board），每个平板上有一堆整齐排列的点，且已知这些平板的放置位置与平板上的点的间距，再指定世界坐标系的原点与坐标轴方向，就能计算出每个点的世界坐标

假设已知图像上的一点 $(x,y)$ 对应的世界坐标为 $(X,Y,Z)$ ，则可列出以下方程

$$
\begin{pmatrix}
x \\
y \\
1
\end{pmatrix}
=K[R\quad t]\begin{pmatrix}
X \\
Y \\
Z \\
1
\end{pmatrix}=
\begin{bmatrix}
p_{11} & p_{12} & p_{13} & p_{14} \\
p_{21} & p_{22} & p_{23} & p_{24} \\
p_{31} & p_{32} & p_{33} & p_{34}
\end{bmatrix}
\begin{pmatrix}
X \\
Y \\
Z \\
1
\end{pmatrix}
$$

>补充一点，涉及齐次坐标的式子中的 “=” 是在齐次坐标意义下的相等，即 $[x\ y\ 1]$ 等价于 $[wx\ wy\ w]$

把它展开可得如下 2 个方程

$$
\begin{align}
x_i p_{31} X_i + x_i p_{32} Y_i + x_i p_{33} Z_i + x_i p_{34} - (p_{11} X_i + p_{12} Y_i + p_{13} Z_i + p_{14}) &= 0\\
y_i p_{31} X_i + y_i p_{32} Y_i + y_i p_{33} Z_i + y_i p_{34} - (p_{21} X_i + p_{22} Y_i + p_{23} Z_i + p_{24}) &= 0
\end{align}
$$

表述为矩阵形式

$$A_i\cdot \mathbf{p} = \mathbf{0}$$

其中

$$
\begin{align}
A_i &=
\begin{pmatrix}
X_i & Y_i & Z_i & 1 & 0 & 0 & 0 & 0 & -x_i X_i & -x_i Y_i & -x_i Z_i & -x_i \\
0 & 0 & 0 & 0 & X_i & Y_i & Z_i & 1 & -y_i X_i & -y_i Y_i & -y_i Z_i & -y_i
\end{pmatrix}\\
\mathbf{p} &= [p_{11}, p_{12}, \dots, p_{34}]^T
\end{align}
$$

在 $\mathbf{p}$ 中共有 12 个未知量，由于采用的是齐次坐标，所以其变为不同倍数仍是等价的，我们规定

$$||p||^2=1$$

故共有 11 个自由度，而一组对应点能提供 2 个方程，则至少需要 6 组对应点，把它们都写在一起，得到以下方程

$$A_{2n \times 12} \cdot \mathbf{p} = \mathbf{0}_{2n}$$

利用最小二乘法求解即可，而后恢复为矩阵 $P$ 

在求出 $P$ 后，需要从中得到内外参，我们已知

$$P=\begin{bmatrix}
p_{11} & p_{12} & p_{13} & p_{14} \\
p_{21} & p_{22} & p_{23} & p_{24} \\
p_{31} & p_{32} & p_{33} & p_{34}
\end{bmatrix}=K[R\quad t]$$

这可以进一步拆分为

$$
\begin{bmatrix}
p_{11} & p_{12} & p_{13}\\
p_{21} & p_{22} & p_{23}\\
p_{31} & p_{32} & p_{33}
\end{bmatrix}=KR\quad(1),\quad
\begin{bmatrix}
p_{14} \\
p_{24} \\
p_{34}
\end{bmatrix}=Kt\quad(2)
$$

对于 $(1)$ 式，其中 $K$ 是上三角矩阵，$R$ 是正交矩阵，这正好与矩阵的 RQ 分解一致（QR 分解：任意矩阵 = 正交阵 · 上三角阵；RQ 分解：任意矩阵 = 上三角阵 · 正交阵）

在使用 RQ 分解得到 $K$ 和 $R$ 后，就可以利用 $(2)$ 式计算出 $t$ 

## QR 分解

矩阵的 QR 分解过程如下，假设待分解的矩阵 $A$ 由一堆列向量构成

$$A = 
\begin{bmatrix}
\mathbf{a}_1 & \mathbf{a}_2 & \cdots & \mathbf{a}_n
\end{bmatrix}$$

把这些列向量先进行施密特正交化，再进行单位化，得到 $e_1 ... e_n$ 

$$
\begin{align}
&\mathbf{u}_1 = \mathbf{a}_1, \quad \mathbf{e}_1 = \frac{\mathbf{u}_1}{\|\mathbf{u}_1\|}\\
&\mathbf{u}_2 = \mathbf{a}_2 - (\mathbf{a}_2 \cdot \mathbf{e}_1)\mathbf{e}_1, \quad \mathbf{e}_2 = \frac{\mathbf{u}_2}{\|\mathbf{u}_2\|}\\
&\mathbf{u}_{k+1} = \mathbf{a}_{k+1} - (\mathbf{a}_{k+1} \cdot \mathbf{e}_1)\mathbf{e}_1 - \cdots - (\mathbf{a}_{k+1} \cdot \mathbf{e}_k)\mathbf{e}_k, \quad \mathbf{e}_{k+1} = \frac{\mathbf{u}_{k+1}}{\|\mathbf{u}_{k+1}\|}
\end{align}
$$

据此就可以对 $A$ 进行分解

$$
\begin{align}
A &=
\begin{bmatrix}
\mathbf{a}_1 & \mathbf{a}_2 & \cdots & \mathbf{a}_n
\end{bmatrix}\\
&=
\begin{bmatrix}
\mathbf{e}_1 & \mathbf{e}_2 & \cdots & \mathbf{e}_n
\end{bmatrix}
\begin{bmatrix}
\mathbf{a}_1 \cdot \mathbf{e}_1 & \mathbf{a}_2 \cdot \mathbf{e}_1 & \cdots & \mathbf{a}_n \cdot \mathbf{e}_1 \\
0 & \mathbf{a}_2 \cdot \mathbf{e}_2 & \cdots & \mathbf{a}_n \cdot \mathbf{e}_2 \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \cdots & \mathbf{a}_n \cdot \mathbf{e}_n
\end{bmatrix}
= QR
\end{align}
$$

在 RQ 分解中，是对 $A$ 的行向量进行正交化和单位化，且是从最后一行开始的

## 非线性标定

上述方法是通过解线性方程组进行相机标定，是线性的标定方式，但解方程的结果并不能直接得到相机内外参，还要进一步通过 RQ 分解得到

实际上，更多的采用的是非线性的标定方式，其定义了一个非线性的目标函数，通过某种优化算法逐渐去逼近相机参数，一般是用某种误差函数

$$\sum_i \|\text{proj}(\mathbf{K}[\mathbf{R} \mid \mathbf{t}]\mathbf{X}_i) - \mathbf{x}_i\|_2^2$$

其中投影函数把得到的齐次坐标转化回图像上的二维坐标

$$
\text{proj}(\mathbf{X}) = 
\begin{pmatrix}
x' \\
y'
\end{pmatrix}, \quad x' = \frac{x}{z}, \quad y' = \frac{y}{z}
$$

目标是找到一个相机参数，使得这个误差最小，这可以利用一些优化算法逐步迭代实现，可以把线性标定得到的结果作为迭代的初始值

非线性的方法往往更加精确，且可以通过改变函数形式应对镜头畸变等复杂情况

## Vanishing Point 标定

有时候并不能获取图像中的点对应的世界坐标，比如给定一个已经拍摄好的照片，要计算拍摄时用的相机的参数，此时可以利用 vanishing point 计算

但这种方法需要对图片有一定要求，即图片中可以得到 3 个正交的 vanishing 方向，以此为坐标轴方向建立世界坐标系

![[CV/img/img5/image-4.png]]

>在上图的例子中，垂直方向的线是平行的，可以看作它们相交于无穷远处，称为 infinite vanishing point ，而另外两个有实际交点的就是 finite vanishing point

前面相机标定中的 $P$ 与 vanishing point 的坐标有以下关联

$$
\begin{align}
&\begin{bmatrix}
p_{11} & p_{12} & p_{13} & p_{14} \\
p_{21} & p_{22} & p_{23} & p_{24} \\
p_{31} & p_{32} & p_{33} & p_{34}
\end{bmatrix}
\begin{pmatrix}
1 \\
0 \\
0 \\
0
\end{pmatrix}
= v_1
\begin{bmatrix}
p_{11} & p_{12} & p_{13} & p_{14} \\
p_{21} & p_{22} & p_{23} & p_{24} \\
p_{31} & p_{32} & p_{33} & p_{34}
\end{bmatrix}
\begin{pmatrix}
0 \\
1 \\
0 \\
0
\end{pmatrix}
= v_2 \\
&\begin{bmatrix}
p_{11} & p_{12} & p_{13} & p_{14} \\
p_{21} & p_{22} & p_{23} & p_{24} \\
p_{31} & p_{32} & p_{33} & p_{34}
\end{bmatrix}
\begin{pmatrix}
0 \\
0 \\
1 \\
0
\end{pmatrix}
= v_3
\begin{bmatrix}
p_{11} & p_{12} & p_{13} & p_{14} \\
p_{21} & p_{22} & p_{23} & p_{24} \\
p_{31} & p_{32} & p_{33} & p_{34}
\end{bmatrix}
\begin{pmatrix}
0 \\
0 \\
0 \\
1
\end{pmatrix}
= O_{\text{world}}
\end{align}
$$

即 $P$ 的前三个列向量 $p_1\ p_2\ p_3$ 就是 $x\ y\ z$ 方向的 vanishing point 在图像中的齐次坐标 $v_i$ ，而最后一列 $p_4$ 是世界坐标系原点在图像中的齐次坐标 $O_{\text{world}}$

此外，这 3 个 vanishing 方向 $e_i$ 是相互正交的，而 $e_i$ 与 $v_i$ 有以下关系

$$
\begin{align}
v_i = P
\begin{pmatrix}
e_i \\
0
\end{pmatrix} = K[R|t]
\begin{pmatrix}
e_i \tag{1}\\
0
\end{pmatrix}
=KRe_i\\
\text{where }
e_1 = 
\begin{pmatrix}
1 \\
0 \\
0
\end{pmatrix}, \quad 
e_2 = 
\begin{pmatrix}
0 \\
1 \\
0
\end{pmatrix}, \quad 
e_3 = 
\begin{pmatrix}
0 \\
0 \\
1
\end{pmatrix}
\end{align}
$$

则有

$$e_i = R^T K^{-1} v_i$$

由相互正交性可得 $e_i^Te_j=0\ (i\neq j)$ ，即

$$v_i^T K^{-T} RR^T K^{-1} v_j = 0$$

而 $R$ 是正交阵，$R^TR=I$ ，故最终得到以下方程

$$v_i^T K^{-T}K^{-1} v_j = 0 \ (i\neq j)\tag{2}$$

其中的 $v_i$ 可以从图像中求得，根据方程 $(2)$ 就能解得内参 $K$ ，再利用方程 $(1)$ 就可以解得 $R$ ，对于 $t$ ，可以根据 $P[0\ 0\ 0\ 1]^T=Kt=p_4$ 得到

对于方程 $(2)$ 的求解，由于 $K$ 的特殊性，可得 $K^{-1}$ 与 $K^{-T}$ 的形式如下，其中 $g = \frac{1}{f}$

$$
K = 
\begin{bmatrix}
f & 0 & c_x \\
0 & f & c_y \\
0 & 0 & 1
\end{bmatrix}, \quad
K^{-1} = 
\begin{bmatrix}
g & 0 & -g c_x \\
0 & g & -g c_y \\
0 & 0 & 1
\end{bmatrix}, \quad
K^{-T} = 
\begin{bmatrix}
g & 0 & 0 \\
0 & g & 0 \\
-g c_x & -g c_y & 1
\end{bmatrix}
$$

令 $v_i = (x_i, y_i, w_i)$ ，则方程 $(2)$ 变为

$$
\begin{align}
v_i^T K^{-T} K^{-1} v_j &= g^2 (x_i x_j + y_i y_j) - g^2 c_x (w_i x_j + w_j x_i) - g^2 c_y (w_i y_j + w_j y_i) \\
&+ g^2 (c_x^2 + c_y^2) w_i w_j + w_i w_j = 0 \\
\end{align}
$$

这种方法的优缺点有
- 优点：不需要手动放置标定板并计算图像上的点对应的世界坐标，能够自动根据图片进行计算
- 缺点：只适用于特定角度拍摄的场景，而且 vanishing point 的精确定位是比较困难的，同时至少需要 2 个 finite vanishing point

![[CV/img/img5/image-5.png]]

# Triangulation

三角测量：已知空间中某点在两个图像中的坐标，以及相机参数，求其世界坐标

![[CV/img/img5/image-1.png]]

由于对应点匹配时存在误差，直接对两个点利用相机成像的逆变换求得的世界坐标不一定完全一致，即图中的两个射线 $O_1x_1$ 和 $O_2x_2$ 不一定相较于一点

## 几何方法

找到两个射线间的最短线段，并把 X 设为其中点

![[CV/img/img5/image-2.png]]

## 非线性方法

设置一个目标误差函数，找到一个 X 使得误差最小 

![[CV/img/img5/image-3.png]]

## 线性方法

已知 $x_i=P_iX$ ，在齐次坐标下，可以看作向量 $x_i$ 与向量 $P_iX$ 成倍数关系，即二者平行，则它们的叉乘为 0 ，这就转化为

$$x_i\times P_iX=0$$

而对于叉乘 $a\times b$ ，也可以用矩阵乘法表示，把 $a$ 变化后的矩阵记为 $[a_\times]$

$$\mathbf{a} \times \mathbf{b} = \begin{bmatrix} a_2b_3 - a_3b_2 \\ a_3b_1 - a_1b_3 \\ a_1b_2 - a_2b_1 \end{bmatrix}= 
\begin{bmatrix}
0 & -a_3 & a_2 \\
a_3 & 0 & -a_1 \\
-a_2 & a_1 & 0
\end{bmatrix}
\begin{bmatrix}
b_1 \\
b_2 \\
b_3
\end{bmatrix}
= [\mathbf{a}_\times]\mathbf{b}
$$

则方程可进一步变为以下形式，这正好是 $AX=0$ 的形式，可以把 $i=1,2$ 对应的方程写到一起，再添加约束 $\lVert X\rVert^2=1$ ，并用最小二乘法求解

$$[{x_i}_\times]P_iX=([{x_i}_\times]P_i)X=0$$










