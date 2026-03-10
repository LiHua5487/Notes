
# 基本概念

假设有两个相机，分别在 $O$ 和 $O'$ 处，成像平面如图，空间中的一点在两个成像平面上的投影点就构成了一组对应点

![[CV/img/img6/image.png]]

- 基线 baseline ：两个相机位置的连线，即 $OO'$ 
- 对极点 epipole ：基线与成像平面的交点
- 对极平面 epipolar plane ：空间中的一点与 $O$ 和 $O'$ 形成的平面
- 对极线 epipolar line ：对极平面与成像平面的交线，也是空间中的点在成像平面上的投影点与对极点的连线

对于两个视角交汇的相机，在各自的成像平面上，所有极线都会相交于极点

![[CV/img/img6/image-1.png]]

值得注意的是极点并不一定处在图像内部，比如以下情况

![[CV/img/img6/image-3.png]]

对于两个平行放置的相机，所有极线都是平行的，可以认为极点处在无穷远处

![[CV/img/img6/image-2.png]]

如果相机是垂直成像平面进行移动的，那两个极点就与 $O$ 和 $O'$ 共线，即处在相机的移动方向上

![[CV/img/img6/image-4.png]]

# Essential and Fundamental Matrix

考虑其中一个图像上的一个点 $x$ ，我们想找到其在另一个图像上的对应点，假设 $x$ 是由空间中一点 $X$ 投影得到的，则 $X$ 应该处在 $Ox$ 射线上，当 $X$ 在射线上移动时，其在另一个图像中的投影点也会沿着一条线移动，这个线就是 $x$ 在另一个图像对应的极线，其对应点一定在这条线上

![[CV/img/img6/image-5.png]]

有了这个性质，在找对应点时，就不需要在另一个图像上全局搜索，只需要在极线上搜索，那怎么根据 $x$ 的坐标计算其对应的极线方程呢？

## Essential Matrix

假设相机内参 $K$ 已知，又知道两个相机位置与朝向之间的变换关系 $R$ 和 $t$ ，把 $x$ 所处图像对应的相机外参设为 $[I\ 0]$ ，则另一个相机的外参就是 $[R\ t]$  

记 $x$ 代表的是 $X$ 在相机坐标系下的坐标，$x_{\text{pixel}}$ 代表在图像中的坐标，另一个图像上的点记为 $x'$ ，则

$$
\begin{align}
&x=K^{-1}x_{\text{pixel}}=[I\ 0]X\\
&x'=K'^{-1}x'_{\text{pixel}}=[R\ t]X
\end{align}
$$

此外还可以得到以下关系

![[CV/img/img6/image-6.png]]

可见 $x$ 与 $x'$ 满足关系 $x'=Rx+t$ ，二者是线性相关的，这可以使用混合积 triple product 来表示

$$x'\cdot [t\times (RX)]=0$$

>向量 $a\ b\ c$ 的混合积 $a\cdot (b\times c)$ 的几何含义是以它们为边的平行六面体的体积，所以可推出以下性质： $a\cdot (b\times c)=c\cdot (a\times b)=b\cdot (c\times a)$

把点乘和叉乘都用矩阵乘表示，可得

$$x'^T[t_\times]Rx=0$$

把 $[t_\times]R$ 定义为一个矩阵 $E_{3\times 3}$ ，这就是 **本质矩阵 essential matrix** 

$$x'^TEx=0$$

- 可计算得 $E$ 秩为 2 
- $R$ 和 $t$ 各 3 个自由度，而 $E$ 在齐次坐标下放缩等价，共 5 个自由度

据此就可以得到极线方程

![[CV/img/img6/image-7.png]]

此外，还可以得到极点，以 $x$ 所处图片的极点 $e$ 为例，由于极点处在所有极线上，可得

$$\forall l,\ l^Te=0$$

即

$$\forall x',\ x'^TEe=0$$

取 $x'$ 为 $(0\ ...\ 1\ ...\ 0)$ ，可得 $Ee$ 的每一个元素都是 0 ，即

$$Ee=0$$

## Fundamental Matrix

对于不知道相机内参的情况，方程变为

$$x'^TEx=x'^T_{\text{pixel}}K'^{-T}EK^{-1}x_{\text{pixel}}=0$$

把 $K'^{-T}EK^{-1}$ 定义为矩阵 $F$ ，这就是 **基本矩阵 fundamental matrix** 

$$x'^T_{\text{pixel}}Fx_{\text{pixel}}=0$$

- 可计算得 $F$ 秩为 2 
- 一个 3×3 矩阵自由度为 9 ，而 $E$ 在齐次坐标下放缩等价，又由于其不满秩，$Det(F)=0$ ，共 7 个自由度

>在计算 $E$ 的自由度时，是从 6 往下减的，这个 6 就已经包括了不满秩的约束，所以只需额外考虑齐次坐标带来的约束

则极线和极点如下

![[CV/img/img6/image-8.png]]

## Eight-Point Algorithm

在求解 $F$ 时，先不考虑秩为 2 这一约束，此时有 8 个自由度，需要 8 组对应点进行计算

$$
\text{Given correspondences } x = 
\begin{pmatrix}
x \\
y \\
1
\end{pmatrix}^T
\text{ and } x' = 
\begin{pmatrix}
x' \\
y' \\
1
\end{pmatrix}^T
\text{Constraint: } x'^T F x = 0
$$

对于一组对应点，可得方程

$$
\begin{pmatrix}
x' & y' & 1
\end{pmatrix}
\begin{bmatrix}
f_{11} & f_{12} & f_{13} \\
f_{21} & f_{22} & f_{23} \\
f_{31} & f_{32} & f_{33}
\end{bmatrix}
\begin{pmatrix}
x \\
y \\
1
\end{pmatrix}
= 0
$$

用矩阵表示为

$$
(x'x, x'y, x', y'x, y'y, y', x, y, 1)
\begin{pmatrix}
f_{11} \\
f_{12} \\
f_{13} \\
f_{21} \\
f_{22} \\
f_{23} \\
f_{31} \\
f_{32} \\
f_{33}
\end{pmatrix}
= 0
$$

可以使用最小二乘法进行求解

初步求解出的 $F$ ，其秩不一定是 2 ，需要通过调整使其满足这一约束，可以对其进行 SVD 分解，把 $\Sigma$ 中奇异值最小的一列去掉（一般是最后一列）

![[CV/img/img6/image-9.png]]

此外，还可以用 SVD 得到两个极点坐标，左右极点分别为 U 和 V 的最后一列

在应用这一约束后，能使计算出的极线都相交于一点

![[CV/img/img6/image-10.png]]

极线的一个很大的用处就是便于找对应点，但计算极线又需要先找到对应点，这看似矛盾，实际上， SIFT 等基本方法检测出的对应点数量往往很有限，可以先用这些方法提取出几组对应点，计算出 $F$ ，这样就能计算图像上任一点在另一个图像中的对应点了，只需要在对应的极线上找到最相似的点即可

---

但是上面的算法有个问题，对于分辨率很大的图，$x$ 和 $y$ 也可能很大，此时乘积项将会非常大，导致数值不稳定，结果误差大

为此，可以对关键点坐标进行归一化，对于一个图片中的一堆关键点，先进行中心化，即取其平均坐标作为中心点，把中心点平移到坐标原点，即每个点 $(x_i, y_i)$ 更新为 $(x_i - x_{\text{mean}}, y_i - y_{\text{mean}})$ 

$$
x_{\text{mean}} = \frac{1}{N} \sum_{i=1}^N x_i, \quad 
y_{\text{mean}} = \frac{1}{N} \sum_{i=1}^N y_i
$$

而后进行放缩，调整点的分布范围，使所有点到原点的平均距离为某个标准值（一般设为 $\sqrt{2}$ 像素）

$$
d_{\text{mean}} = \frac{1}{N} \sum_{i=1}^N \sqrt{x_i^2 + y_i^2}
$$

设放缩倍数为 $s$ ，则放缩后平均距离 $sd_{\text{mean}}=\sqrt2$ ，则

$$
s = \frac{\sqrt{2}}{d_{\text{mean}}}
$$

进一步更新坐标为 $(sx_i,sy_i)$

如果采用齐次坐标，则上述归一化可以用矩阵表示

$$T = 
\begin{bmatrix}
s & 0 & -s \cdot x_{\text{mean}} \\
0 & s & -s \cdot y_{\text{mean}} \\
0 & 0 & 1
\end{bmatrix}$$

假设两个图的归一化矩阵分别为 $T$ 和 $T'$ ，在使用归一化后的坐标计算 $F$ 后，需要再变回来

$$F'=T'^TFT$$

---

此外，还可以用非线性的方式计算 $F$ 

![[CV/img/img6/image-11.png]]

---

最后，让我们来听一曲 [The Fundamental Matrix Song](http://danielwedge.com/fmatrix/) 吧（查询创作者的精神状态


