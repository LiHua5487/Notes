
# Photometric Stereo

Photometric Stereo：在一个拍摄角度下（不需要相机参数），应用不同方向的光照去还原物体的 3D 表面

![[CV/img/img10/image-1.png]]

这是因为，受到不同方向的光照时，物体的局部表面会反射出不同的颜色，这可以通过 **BRDF (Bidirectional Reflectance Distribution Function)** 确定，其描述了当光线从一个方向照射到表面时，从另一个方向观察时其亮度是多少

$$BRDF = f(\theta_i, \phi_i, \theta_e, \phi_e)$$

为了简化问题，假设光在物体表面只发生**漫反射 diffuse reflection** ，这意味着从不同角度去观察一个局部表面，其颜色都是一样的（如果存在镜面等特殊表面，情况就不同了）

![[CV/img/img10/image.png]]

根据 **Lambert’s Law** ，一个局部表面在受到一个方向的光照射后，其反射出的光强 $I$ 如下，可见在物体和光照确定的情况下，不同位置的光强只与其表面朝向有关

$$I = \rho (n \cdot s) = \rho \|s\| \cos \theta$$

- $\rho$ ：反射率
- $n$ ：表面的单位法向量
- $s$ ：入射光方向，与 $n$ 夹角为 $\theta$ ，大小与入射光强成正比

假设一个平面的方程为 $z=f(x,y)$ ，则其表面法向量为（未归一化）

$$N = \left( -\frac{\partial z}{\partial x}, -\frac{\partial z}{\partial y}, 1 \right) \coloneqq (p, q, 1)$$

为了直观的展示不同表面朝向对应的光强，引入 **Gradient Space** ：取坐标系中 $z=1$ 的平面，原点到平面上某点的连线朝向就代表一个表面的法向量，这个位置的强度值就对应着这个表面的光强 $I$ 

![[CV/img/img10/image-2.png]]

这里在计算 $I$ 时将 $n$ 和 $s$ 归一化，即

$$
\begin{align}
n = \frac{N}{\|N\|} = \frac{(p, q, 1)}{\sqrt{p^2 + q^2 + 1}}\\
s = \frac{S}{\|S\|} = \frac{(p_s, q_s, 1)}{\sqrt{p_s^2 + q_s^2 + 1}}
\end{align}
$$

则

$$
\begin{align}
I &= \rho (n \cdot s) \\
  &= \rho \frac{pp_s + qs + 1}{\sqrt{p^2 + q^2 + 1} \sqrt{p^2_s + q^2_s + 1}} \\
  &\coloneqq R(p, q)
\end{align}
$$

即在 $\rho$ 和 $s$ 均已知的情况下，$I$ 就可以视为关于 $p$ 和 $q$ 的函数 $R(p,q)$ ，则 Gradient Space 的图像如下（称为 **Reflectance Map**） ，其中亮度最大的点代表法向量和光照方向相同

![[CV/img/img10/image-3.png]]

由公式可知，所有与 $s$ 成相同夹角的 $n$ 都对应着同一光强，可以画出一系列等光强轮廓线，对于物体表面上一个局部区域，从拍摄的图像中可以获取其强度值，再到 reflectance map 中查找就能得到其法向量，但这显然是不唯一的

![[CV/img/img10/image-4.png]]

为此，需要应用不同方向的光照拍摄多张图像，以确保解的唯一性，对于漫反射场景，最少需要 3 张图像，其它情况可能更多

![[CV/img/img10/image-5.png]]

在求解表面法向量时，可以列出以下方程

$$I_1 = \rho s_1 \cdot N, \quad I_2 = \rho s_2 \cdot N, \quad I_3 = \rho s_3 \cdot N$$

写成矩阵形式

$$
\begin{bmatrix}
I_1 \\
I_2 \\
I_3
\end{bmatrix} = \rho 
\begin{bmatrix}
s_{x_1} & s_{y_1} & s_{z_1} \\
s_{x_2} & s_{y_2} & s_{z_2} \\
s_{x_3} & s_{y_3} & s_{z_3}
\end{bmatrix} N
\coloneqq \rho SN
$$

这里要求 $N$ 是单位向量，我们可以先不管 $\rho$ ，求解 $I=SN$ ，在得到 $N$ 后，需要对其进行归一化

$$\hat{N} = \frac{N}{\lVert N\rVert} $$

而后令 $\rho = \lVert N\rVert$ 即可，相当于 $I=SN=\rho S\hat{N}$

如果 $S$ 可逆，那就能直接利用矩阵的逆计算出 $N$ ；但是 $S$ 可能不可逆，这对应着 $s_1$ $s_2$ $s_3$ 共面

而且，我们一般会拍摄不只 3 张图像，即

$$
\begin{bmatrix}
I_1 \\
I_2 \\
\vdots \\
I_K
\end{bmatrix} = \rho 
\begin{bmatrix}
s_{x_1} & s_{y_1} & s_{z_1} \\
s_{x_2} & s_{y_2} & s_{z_2} \\
\vdots & \vdots & \vdots \\
s_{x_K} & s_{y_K} & s_{z_K}
\end{bmatrix} N
\coloneqq \rho SN
$$

这正好是 $AX=B$ 的形式，可以用最小二乘法求解（与前面相同，先不管 $\rho$ ，求解 $I=SN$ ，再归一化）

## Calibration based Photometric Stereo

上述解方程的方法是针对漫反射情况的，对于一般情况，需要利用一个校准物体（具有已知的大小、形状，且表面和待测物体一样，一般是一个球）获取 reflectance map ，再根据拍摄的图像从中查找对应的法向量

这里需要假设 **Orientation Consistency** ：反射只依赖于法向和光照方向，而不依赖于其他因素（但是具体的强度值可能不是 Lambert’s Law 的点积形式）

先用一系列光照方向 $\{s_1,s_2,\cdots,s_n\}$ 分别照射标定物体，对于一个局部表面，其对应着一系列强度值 $\{I_1,I_2,\cdots,I_n\}$ ，而由于物体形状已知，其法向量 $n$ 是可以算出来的，可以建立一个查找表，根据强度序列确定法向量

![[CV/img/img10/image-6.png]]

而后应用相同的光照方向 $\{s_1,s_2,\cdots,s_n\}$ 去拍摄待测物体，每个点得到一系列强度值，根据查找表就可以确定其法向量

![[CV/img/img10/image-7.png]]

## Shape from Normals

得到一堆法向量后，要还原出表面的具体位置，就需要计算图像中每个像素 $(x,y)$ 的深度值 $z$ ，我们知道表面位置和法向量互为微分与积分的关系

![[CV/img/img10/image-8.png]]

假设表面 $z=f(x,y)$ 是连续可微的，这意味着 $z$ 存在全微分

$$dz = \frac{\partial z}{\partial x} dx + \frac{\partial z}{\partial y} dy = -p dx - q dy$$

且表面上两点间的积分与路径无关

$$\int_{(x_0,y_0)}^{(x,y)} \frac{\partial z}{\partial x} dx + \frac{\partial z}{\partial y} dy = \int_{(x_0,y_0)}^{(x,y)} -p dx - q dy = z(x,y) - z(x_0,y_0)$$

我们就可以从一个参考点 $(x_0,y_0)$ 出发，计算其他各个点的位置，且由于与路径无关，我们可以选择一个相对简单的路径

![[CV/img/img10/image-9.png]]

考虑每一步移动，以 $(x,y-1)$ 到 $(x,y)$ 为例，将上述公式离散到像素，就是

$$z(x,y)=z(x,y-1)-p(x,y)\Delta x-q(x,y)\Delta y$$

而 $\Delta x=0,\Delta y=1$ ，故

$$z(x,y)=z(x,y-1)-q(x,y)$$

在计算时，先设置好参考点 $z(0,0) = 0$ ，而后计算第一列的值

```
for y = 1 to (H - 1)
	z(0,y) = z(0,y - 1) - q(0,y)
```

再计算每一行的值

```
for y = 1 to (H - 1)
   for x = 1 to (W - 1)
	   z(x,y) = z(x - 1,y) - p(x,y)
```

由于法向量的计算是存在噪音的，所以不同路径的计算结果不一定相同，可以取不同路径的计算结果的均值

![[CV/img/img10/image-10.png]]

---

另一种方法是 Poisson最小二乘法，我们希望重建表面的梯度值 $\frac{\partial z}{\partial x}$ 接近 $-p$ ，$\frac{\partial z}{\partial y}$ 接近 $-q$ ，为此定义以下误差函数

$$D = \iint_{Image} \left( \frac{\partial z}{\partial x} + p \right)^2 + \left( \frac{\partial z}{\partial y} + q \right)^2 dx dy$$

离散到像素上就是

$$D = \sum_{x} \sum_{y} \left\| \nabla z(x,y) + \begin{bmatrix} p \\ q \end{bmatrix} \right\|^2$$

这与之前的 Poisson image blending 类似，可以采用相同的方法求解

# Shape from Shading

Shape from Shading 的目标是已知光照方向、物体的反射模型，从一张图像中恢复 3D 表面

但我们只有一个图像，每个像素的亮度方程是一个方程、两个未知数（表面法向量的两个分量 $p$ 和 $q$），这是一个欠定问题，无法直接求解，需要依赖全局优化，将表面表示为梯度空间 $(p, q)$ ，但 Gradient Space 有奇点问题（当表面垂直 $z$ 轴时，$p$ 或 $q$ 趋于无穷大），导致数值计算不稳定，所以需要对 Gradient Space 进行改进

我们不从原点出发，而是从 $(0,0,-1)$ 出发，具体情况如图

![[CV/img/img10/image-13.png]]

这样所有可能的情况都落在一个半径为 2 的圆里（夹角为钝角说明被遮挡了，根本看不见，所以不考虑），可以表示为 $I=R(f,g)$ 

![[CV/img/img10/image-14.png]]

由于我们已知反射模型，就可以利用公式计算出 reflectance map ，而不需要实际测量

---

但还存在歧义性的问题，比如以下情况，左图看上去是个坑，但是如果翻转 180° （右图），看上去像是凸起的

![[CV/img/img10/image-11.png]]

光照方向不同也会带来歧义性，比如下面的人脸是往外凸还是往里凹的

![[CV/img/img10/image-12.png]]

所以需要引入额外约束

**Occluding Boundaries Constraint** ：在观察到的物体边缘处，表面法向量 $n$ 、边缘切向量 $e$ 、观察角度 $v$ 三者应该相互垂直，且满足以下的相对关系

![[CV/img/img10/image-15.png]]

**Image Intensity Constraint** ：图像上每个像素的亮度应该符合 reflectance map ，即 $I(x,y) = R_s(f,g)$ ，可以定义一个误差项 data term 

$$e_r = \iint (I(x,y) - R_s(f,g))^2 dx dy$$

**Smoothness Constraint** ：物体表面应该光滑，即梯度变化不希望过快，定义另一个误差项 smooth term

$$e_s = \iint (f_x^2 + f_y^2) + (g_x^2 + g_y^2) dx dy$$

- $f_x = \frac{\partial f}{\partial x}, \quad f_y = \frac{\partial f}{\partial y}, \quad g_x = \frac{\partial g}{\partial x}, \quad g_y = \frac{\partial g}{\partial y}$

把后两个约束整合一下，需要最小化下面的误差

$$e =e_s +\lambda e_r$$

而从 Occluding Boundaries Constraint 可以得到边缘处的 $(f,g)$ 值，将其作为常量













