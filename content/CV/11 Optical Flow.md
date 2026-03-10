
# Optical Flow

当我们看一个视频时，能轻易分辨出物体是如何移动的，但是怎么让计算机去理解呢？当一个物体运动时，图像上一些位置的像素颜色和亮度就会发生变化，光流 Optical Flow 的核心思想是，通过分析相邻帧之间像素点的亮度变化，来估计出每一个像素在图像平面上的瞬时速度，进而反映物体的运动

但是光流并不一定能准确反映运动，比如一个均匀的球体在自转，如果光照不变，那其表面各处的亮度就不变，也就没有光流；或者物体不动，但外界光照发生变化，导致物体表面亮度改变，那就有了光流

此外，从光流中推测的运动方向也不一定就是实际的运动方向，比如一个旋转的灯带，实际上是横向旋转运动的，但是视觉上看来，这些红蓝颜色条带像是在竖直的移动

![[CV/img/img11/image.png]]

## Lucas-Kanade Optical Flow

假设有一个输入的视频，即一系列图像帧，给每一帧打一个时间标签 $t$ ，则可以用下面的变量来描述一个像素的强度值

![[CV/img/img11/image-1.png]]

光流就是要求解出从 $t$ 时刻到 $t+1$ 时刻，每个像素的运动矢量，这可以看作一个对应点匹配问题，即在 $t+1$ 时刻的图像中找到 $t$ 时刻图像中的一个像素的对应点，考虑到两个时刻间隔非常短，**Lucas-Kanade Optical Flow** 方法中做出一了些假设
- 颜色/亮度恒定性：一个点在两帧中的颜色应该基本相同
- 小位移：一个点不会移动很远
- 运动一致性：一个点与其邻近点的移动方式应该差不多

假设一点从 $(x,y)$ 移动到了 $(x+u,y+v)$ ，我们要求解出 $(u,v)$ ，直接暴力匹配开销太大了，考虑先列一些方程，由亮度恒定性，有

$$I(x,y,t)=I(x+u,y+v,t+1)$$

将右式用泰勒展开近似可得

$$I(x + u,y + v,t + 1) \approx I(x,y,t) + I_x u + I_y v + I_t$$

其中 $I_x=\frac{\partial I}{\partial x},I_y=\frac{\partial I}{\partial y}$  ，则

$$I_x u + I_y v + I_t=\nabla I[u,v]+I_t \approx 0$$

当 $[u,v]$ 比较小时，可以这么近似，但是如果 $u$ 或 $v$ 非常大，就不行了，所以需要“小位移”的假设

将 $I_x u + I_y v + I_t=0$ 视为一个直线，则点 $(x,y)$ 对应的光流 $(u,v)$ 就处在这个直线上

![[CV/img/img11/image-2.png]]

将 $(u,v)$ 分解为沿着直线 $u_p$ 和垂直直线 $u_n$ 这两个分量，很容易计算出 $u_n$ 

$$u_n = \frac{|I_t|}{(I_x^2 + I_y^2)}(I_x, I_y)$$

但 $u_p$ 没法确定，这就带来歧义性，所以说从光流推断的运动方向和真实的运动方向不一定一样

为此，利用“运动一致性”的假设，取一个 5×5 的小窗口，假设其光流一样，即 $(u,v)$ 相同，则每一个像素都可以列一个方程，整合到一起，发现正好是 $Ax=b$ 的形式

$$
\underbrace{\begin{bmatrix}
I_x(p_1) & I_y(p_1) \\
\vdots & \vdots \\
I_x(p_{25}) & I_y(p_{25})
\end{bmatrix}}_{A}
\underbrace{\begin{bmatrix}
u \\
v
\end{bmatrix}}_{x}
=
\underbrace{-\begin{bmatrix}
I_t(p_1) \\
\vdots \\
I_t(p_{25})
\end{bmatrix}}_{b}
$$

利用最小二乘法，即求解 $A^TAx=A^Tb$ ，也就是下面的方程

$$
\underbrace{\begin{bmatrix}
\sum I_x I_x & \sum I_x I_y \\
\sum I_y I_x & \sum I_y I_y
\end{bmatrix}}_{A^TA}
\begin{bmatrix}
u \\
v
\end{bmatrix}
=
\underbrace{-\begin{bmatrix}
\sum I_x I_t \\
\sum I_y I_t
\end{bmatrix}}_{A^Tb}
$$

左边这个矩阵正好与 Harris Corner 里 response 对应的矩阵一样

## Coarse-to-Fine Optical Flow Estimation

理论上来讲，由于“小位移”的假设，求解出来的光流应该都比较小

![[CV/img/img11/image-3.png]]

但是如果场景中有一些运动很快的物体，这个假设就失效了，求解结果中也会存在一些比较大的光流

注意到，如果我们降低图像的分辨率，那原先比较明显的移动就变得不那么明显了，所以可以构建一个图像金字塔，最顶层是经过多次下采样的、模糊的缩略图，对这一层可以得到一个初步的估计结果，然后将其传递到金字塔的下一层，以其为基础，去估计更精细的运动残差，逐层迭代，这种方法称为 **Coarse-to-Fine Optical Flow Estimation** 

## Horn-Schunck Optical Flow

Horn-Schunck Optical Flow 做出的假设是
- 亮度恒定性：与 Lucas-Kanade Optical Flow 中的相同
- 光滑性：光流向量在空间上是平滑变化的

其 Energy 函数如下

$$
\min_{u,v} \sum_{i,j} \left\{ E_d(i,j) + \lambda E_s(i,j) \right\} = \iint \underbrace{\left( I_x u + I_y v + I_t \right)^{2}}_{\text{brightness constancy}} + \lambda \underbrace{\left( \|\nabla u\|^{2} + \|\nabla v\|^{2} \right)}_{\text{smoothness}} dxdy
$$

其中 smooth term 定义为与周围 4 个像素的光流的差异

$$E_s(i,j) = \frac{1}{4} \left[ (u_{ij} - u_{i+1,j})^2 + (u_{ij} - u_{i,j+1})^2 + (v_{ij} - v_{i+1,j})^2 + (v_{ij} - v_{i,j+1})^2 \right]$$

Lucas-Kanade Optical Flow 中考察一个局部窗口内的光流，适合捕捉细节的运动；Horn-Schunck Optical Flow 考虑全局的光流变化，侧重整体的运动





























