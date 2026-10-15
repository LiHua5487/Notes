
# 1 Phong Illumination

## Phong / Blinn-Phong

Phong 模型由三部分组成：环境光、漫反射光、镜面反射光，总的光照效果是这三部分加起来
- 环境光

$$
I_{\text{ambient}} = k_a \times I_a
$$

	其中 $k_a$ 是环境光系数， $I_a$​ 是环境光强度
- 漫反射光

$$
I_{\text{diffuse}} = k_d \times I_l \times (N \cdot L)
$$

	其中 $k_d$ 是漫反射系数，$I_l$ 是光源强度，$N$ 是表面法向量，$L$ 是光源方向
- 镜面反射光

$$
I_{\text{specular}} = k_s \times I_l \times (R \cdot V)^n
$$

	其中 $k_s$ 是镜面反射系数，$I_l$ 是光源强度，$R$ 是光线反射方向量（光线从表面反射出去的方向），$V$ 是视线方向量（从表面到眼睛的方向），$n$ 是光泽度（越大越光滑）

Blinn-Phong 对镜面反射的部分进行了优化，用半角向量 $H$ 代替反射向量 $R$ 

$$

I_{specular} = k_s \times I_l \times (N \cdot H)^n

$$

其中 $H$ 是半角向量，它是 $L$ 和 $V$ 的对角线方向，并进行归一化

$$

H = \frac{L + V}{\| L + V \|}

$$

用两种方式渲染得到的结果如下

![[VCL/作业/img/img3/image-1.png]]

## Q1

顶点着色器负责处理每个顶点的数据，包括位置变换、光照计算等；片段着色器负责计算每个片段（像素）的最终颜色，包括光照、纹理、透明度等

顶点着色器的输出变量会被用来组成基本的图元，如点、线、三角形等，而后通过光栅化插值，转化为一堆片段，每个片段对应屏幕上的一个像素，再传递到片段着色器

![[VCL/作业/img/img3/image.png]]

>此处参考 [深入解析 OpenGL 着色器：顶点着色器与片段着色器](https://blog.csdn.net/wang1290865309/article/details/150924763)

## Q2

`if (diffuseFactor.a < .2) discard;` 的意思是：当纹理的 alpha 值小于 0.2 时，完全丢弃该片段（不渲染）；当 alpha 值大于等于 0.2 时，正常渲染

由于浮点数计算的误差，`diffuseFactor.a` 很可能不是严格等于 0 的，而且使用阈值可以实现更自然的边缘变化，所以一般不直接用等于号来判断

## Bonus

凹凸映射 Bump Mapping 通过调整物体表面不同地方的明暗程度，使物体看起来有凹凸不平的细节（但表面形状没有改变），这可以通过调整表面的法向量实现

我们先计算原先的法向量，假设采用 $(u,v)$ 坐标，则表面上一点可以表示为 

$$
P=(x(u,v),y(u,v),z(u,v))
$$

分别对 $u$ 和 $v$ 求偏导得 $P_u$ 和 $P_v$ ，则该点法向量可以表示为

$$
N=P_u\times P_v,\quad \hat N=\frac{N}{\|N\|}
$$

而后需要调整法向量，《Bump Mapping Unparametrized Surfaces on the GPU》中的方法是接收一个高度图作为输入，每点有一个高度值 $H$ ，表示沿着原先的法向量方向移动多少（可正可负），则移动后点坐标变为

$$
P'=P+H\cdot \hat{N}
$$

对其求偏导得

$$
P'_u=P_u+H_u\cdot \hat{N}+H\cdot\frac{\partial\hat N}{\partial u}
$$

其中 $\frac{\partial\hat N}{\partial u}$ 是一个高阶小量，可以忽略，则

$$
P'_u=P_u+H_u\cdot \hat{N}
$$

对于 $P'_v$ 同理，可得调整后的法向量为

$$
\begin{aligned}
N'&=P'_u\times P'_v\\
&=P_u \times P_v + H_v \cdot P_u \times \hat N + H_u \cdot \hat N \times P_v + H_u H_v \cdot \hat N \times \hat N\\
&=N+\underbrace{(H_v\cdot P_u\times \hat N-H_u\cdot P_v\times \hat N)}_D \\
\hat N'&=\frac{N'}{\|N'\|}
\end{aligned}
$$

可见我们只需计算 $D$ ，再加到原先的法向量上，最后归一化即可

在计算 $H_u$ 和 $H_v$ 时，直接使用差分近似即可；在计算 $P_u$ 和 $P_v$ 时，OpenGL 提供了 $dFdx$ 和 $dFdy$ 函数，但它们是对屏幕空间 $(x,y)$ 的导数，不过我们可以列出以下方程

$$\begin{cases}
\frac{\partial P}{\partial x}= \frac{\partial P}{\partial u}\frac{\partial u}{\partial x}+\frac{\partial P}{\partial v}\frac{\partial v}{\partial x}\\
\frac{\partial P}{\partial y}= \frac{\partial P}{\partial u}\frac{\partial u}{\partial y}+\frac{\partial P}{\partial v}\frac{\partial v}{\partial y}
\end{cases}$$
而 $\frac{\partial u}{\partial x}$ 这些项就可以采用差分近似，取屏幕上相邻像素的 $(u,v)$ 坐标做差，解方程即可得 $P_u$ 和 $P_v$ 

最后得到的结果如下，bump 程度为 30% ，可见墙上和红色地毯上有了更多的凹凸质感

![[VCL/作业/img/img3/image-2.png]]

# 2 Environment Mapping

立方体贴图就是给立方体的每个面都搞一个 2D 纹理，可以用其来实现天空盒；而后通过计算视线方向的反射向量来采样环境贴图，这只需看向量与天空盒上哪个面相交，并根据交点坐标进行插值处理即可，最后效果如下

![[VCL/作业/img/img3/image-3.png]]

# 3 Non-Photorealistic Rendering

## 轮廓线的渲染

为了得到非真实感的渲染效果，一个简单的算法是给物体搞个轮廓线，并且表面有着从冷色到暖色的过渡

为了得到轮廓线，可以先渲染物体的背面，并稍微扩大一点，后续只需要再渲染出模型的正面，就能实现轮廓线的效果

在模型渲染到屏幕上时，经过了如下的坐标变换

$$

\begin{aligned}
&\text{模型坐标系} \xrightarrow{\text{Model Matrix}} \text{世界坐标系} \xrightarrow{\text{View Matrix}} \text{相机/视图坐标系} \\
&\xrightarrow{\text{投影矩阵 MVP}} \text{裁剪空间} \xrightarrow{\text{透视除法}} \text{归一化设备空间 NDC} \rightarrow \text{屏幕空间}
\end{aligned}

$$

![[VCL/作业/img/img3/image-4.png]]

其中裁剪空间是为了判断模型的哪些部分处在视线之内，而对于”看不见“的部分就不需要渲染了，其中”看得见“的部分对应视锥中两个远近平面之间，代表最大/最小渲染距离

其通过一个投影矩阵，把视锥这个棱台形状的空间变换为立方体形状的空间，变换后”看得见“的部分的坐标满足 $-w < x < w,-w < y < w,-w < z < w$

![[VCL/作业/img/img3/image-5.png]]

顶点渲染器的输出就是在裁剪空间上的，接着由 GPU 自己做透视除法（x、y、z 坐标都除以 w）将顶点转到 NDC ，则 $-1 < x < 1,-1 < y < 1,-1 < z < 1$

>此处参考 [【Unity3D】空间和变换_unity 模型空间到裁剪空间](https://blog.csdn.net/m0_37602827/article/details/129036096)

设模型空间中的顶点位置为 $p$，法向量为 $n$ ，经过视图变换 $V$ 后坐标为
$$

p_{eye} = V \cdot p

$$
又经过投影矩阵 $P$ 变换到裁剪空间，坐标变为
$$

p_{clip} = P \cdot p_{eye} = P \cdot V \cdot p

$$
我们需要在屏幕空间中获得固定宽度的轮廓线，设期望的轮廓线宽度为 $w$ 个像素，屏幕分辨率为 $W \times H$ 

屏幕空间到 NDC 的映射为
$$

x_{ndc} = \frac{2 x_{screen}}{W} - 1, \quad y_{ndc} = 1 - \frac{2 y_{screen}}{H}

$$
因此，一个像素在NDC中的大小为
$$

\Delta x_{ndc} = \frac{2}{W}, \quad \Delta y_{ndc} = \frac{2}{H}

$$
现在考虑顶点偏移，我们希望顶点沿法线方向偏移，但需要在裁剪空间中进行，法线在裁剪空间中的方向为
$$

n_{clip} = (P \cdot V)_{3 \times 1} \cdot n

$$
为了获得屏幕空间中的固定宽度 $w$，先考虑在 NDC 中的偏移量，由于屏幕是 2D 的，我们只需在 xy 平面内进行偏移，其中 $n_{clip, xy}$ 为 $n_{clip}$ 的 xy 分量
$$

\Delta p_{ndc} = w \cdot \left(\frac{2}{W}, \frac{2}{H}\right) \cdot \frac{n_{clip, xy}}{\|n_{clip, xy}\|}

$$
而后考虑在裁剪空间中的偏移量，有
$$

p_{ndc} = \frac{p_{clip} + \Delta p_{clip}}{p_{clip, w}} = \frac{p_{clip}}{p_{clip, w}} + \Delta p_{ndc}

$$
可得裁剪空间中的偏移量为
$$

\Delta p_{clip} = p_{clip, w} \cdot \Delta p_{ndc}

$$
代入 $\Delta p_{ndc}$ 的表达式得
$$

\Delta p_{clip} = p_{clip, w} \cdot w \cdot \left(\frac{2}{W}, \frac{2}{H}\right) \cdot \frac{n_{clip, xy}}{\|n_{clip, xy}\|}

$$

## 正面颜色渲染

Gooch Shading 用冷暖色的变化来实现非真实感渲染，为了计算每一块的颜色，首先需要计算一个在传统 Lambert 光照模型中也使用的系数：表面法向量 $n$ 与光源方向 $l$ 的点积，这代表光线强度
$$

dot = n \cdot l

$$
但是 $dot$ 的取值范围是 $[-1, 1]$ ，需要变换到 $[0,1]$ 作为混合系数

$$

t= \frac{1}{2}(dot+1)

$$

最后通过插值计算颜色，其中 $k_{cool}$ 和 $k_{warm}$ 是事先设定的冷色和暖色

$$

color = (1 - t) \cdot k_{cool} + t \cdot k_{warm}

$$

为了增强卡通效果，我还对 $t$ 进行了离散化处理，这样能产生一些清晰的分界线，最终效果如下

![[VCL/作业/img/img3/image-6.png]]  

## Q1

渲染背面（白色轮廓线）的部分如下

```cpp
glCullFace(GL_FRONT);
glEnable(GL_CULL_FACE);
for (auto const & model : _sceneObject.OpaqueModels) {
    auto const & material = _sceneObject.Materials[model.MaterialIndex];
    model.Mesh.Draw{ _backLineProgram.Use() };
}
```

- `glCullFace(GL_FRONT)` ：剔除正面，只渲染背面
- `glEnable(GL_CULL_FACE)` ：启用面剔除

渲染正面（Gooch着色）的部分如下

```cpp
glCullFace(GL_BACK);
glEnable(GL_DEPTH_TEST);
for (auto const & model : _sceneObject.OpaqueModels) {
    auto const & material = _sceneObject.Materials[model.MaterialIndex];
    model.Mesh.Draw{ material.Albedo.Use(), material.MetaSpec.Use(), _program.Use() };
}
```

- `glCullFace(GL_BACK)` ：剔除背面三角形，只渲染正面三角形
- `glEnable(GL_DEPTH_TEST)` ：启用深度测试，确保正确的深度遮挡
- `_program` ：Gooch 着色的主着色器程序
- 同时使用材质纹理 `Albedo` 和 `MetaSpec` 

## Q2

如果沿着法线移动固定距离，由于还需经过投影变换才能到屏幕上，这就导致投影后的扩展宽度不一致，如果是正交投影相机还好说，如果是透视投影就达不到预期的效果

# 4 Shadow Mapping

## 有向光源与点光源阴影

有向光源的阴影很好判断：以光的位置为视角进行渲染，能看到的东西都将被点亮，看不见的一定是在阴影之中了，其过程如下
- 从光源视角渲染整个场景，只记录深度信息
- 使用正交投影矩阵将场景变换到光源坐标系
- 在片段着色器中，丢弃掉 alpha 值过低的片段，我们认为这些片段太透明了，不会产生阴影
- 最终得到一张从光源视角看去的深度纹理，即阴影贴图

点光源的阴影稍微复杂一点：需要记录下从光源位置看出去，离得最近的物体表面都在哪里，任何比这个“最近表面”更远的东西，自然就在阴影里了
- 一个点光源会向所有方向发光，无法用一张普通的 2D 图片来记录所有方向的深度信息，可以使用一个立方体贴图来记录
- 为了生成这张立方体阴影贴图，需要分别在 +X, -X, +Y, -Y, +Z, -Z 这 6 个方向进行渲染
- 进行 alpha 检测，丢掉透明片段

## Apply Shadow

应用阴影时，需要确定一个点有没有被遮挡，用光源到这个点方向向量作为坐标，从阴影贴图中采样得到最近深度，与这个点到光源的距离（深度）比较，如果大于最近深度，说明被遮挡

代码中还添加了 bias 项，这是为了应对阴影痤疮现象，由于阴影贴图的深度值是离散采样的，存在误差，表面上的点可能错误地被认为比阴影贴图中的深度值更大，导致表面出现不规则的条纹状自阴影

最终结果如下，这里使用 Blinn-Phong ，如果用 Phong 背景中会有大片黑色阴影

![[VCL/作业/img/img3/image-7.png]]
  
## Q1

有向光源使用正交投影，因为有向光源是平行光；点光源使用透视投影，因为点光源是从一个点发散出去的，类似于相机的投影视锥

## Q2

深度值在前面已经预先计算好并存储为阴影贴图了，后续只需从中采样即可

# 5 Whitted-Style Ray Tracing

## Ray/Triangle Intersection

传统的方法是，求射线与三角形所在平面的交点，然后看在不在三角形内部，而 《Fast, Minimum Storage Ray/Triangle Intersection》中的 Möller-Trumbore 算法的过程如下

对于射线，可以用起点 $O$ 和方向 $D$ 来表示

$$

R(t) = O + tD

$$

而三角形可以用重心坐标来表示，设三个顶点为 $V_0$ $V_1$ $V_2$ ，则

$$

T(u,v) = (1 - u - v)V_0 + uV_1 + vV_2

$$

要求 $u \geq 0, v \geq 0$ ，且 $u + v \leq 1$ ，以保证点在三角形内部

如果射线与三角形相交，则射线上某点 $R(t)$ 应该等于三角形上某点 $T(u,v)$ 

$$

O + tD = (1 - u - v)V_0 + uV_1 + vV_2

$$

把未知数 $t, u, v$ 放到一边，变形为

$$

\begin{bmatrix}
-D, & V_1 - V_0, & V_2 - V_0
\end{bmatrix}
\begin{bmatrix}
t \\
u \\
v
\end{bmatrix}
= O - V_0

$$

定义 $E_1 = V_1 - V_0,\ E_2 = V_2 - V_0,\ T = O - V_0$，则方程变为

$$

\begin{bmatrix}
-D, & E_1, & E_2
\end{bmatrix}
\begin{bmatrix}
t \\
u \\
v
\end{bmatrix}
= T

$$

这可以用克拉默法则来求解，结果为

$$

\begin{bmatrix}
t \\
u \\
v
\end{bmatrix}
=
\frac{1}{(D \times E_2) \cdot E_1}
\begin{bmatrix}
(T \times E_1) \cdot E_2 \\
(D \times E_2) \cdot T \\
(T \times E_1) \cdot D
\end{bmatrix}

$$

在实际计算中，常常先计算一些中量来加速，比如定义 $P = D \times E_2$ 和 $Q = T \times E_1$，则结果为

$$

\begin{bmatrix}
t \\
u \\
v
\end{bmatrix}
=
\frac{1}{P \cdot E_1}
\begin{bmatrix}
Q \cdot E_2 \\
P \cdot T \\
Q \cdot D
\end{bmatrix}

$$

计算出 $t, u, v$ 后，需要检查它们是否有效
- $t\geq 0$ ：交点在射线的正方向上
- $u \geq 0,\ v \geq 0,\ u + v \leq 1$ ：交点在三角形内部

## Whitted Ray Trace

Whitted 风格的光追算法过程如下
- 从相机位置向一个像素发射若干条光线（由采样率控制）
- 计算这条光线是否与场景中的物体相交，并找到最近的交点
- 计算交点颜色，由直接光照、反射、折射三部分组成
- 每个光线最多反射/折射一定次数（由最大深度控制）
- 将所有交点的颜色混合，得到像素的最终颜色

下面是采样率为 1 ，最大深度为 3 情况下的渲染效果

![[VCL/作业/img/img3/image-8.png]]

## Q1

光栅化的结果相比光追少了一些细节，因为光栅化赋予颜色时只考虑光源对一点的影响，没有考虑到其它地方的反射光和折射光的影响；而光追是模拟光线的传播途径，更加真实

## Bonus

我搞了个 BVH 来加速，其想法是把场景中的物体分组放进不同的盒子里（层次包围盒），然后再把这些小盒子放进更大的盒子里，形成一个树结构，对于一个光线，可以先检查它是否击中大盒子，如果没击中，就可以跳过整个大盒子里的所有物体，大大减少计算量

这里我采用 AABB 包围盒 (Axis-aligneded Bounding Box) ，即一个与坐标轴对齐的立方体，只需记录最大和最小的两个顶点就可以表示其范围，内部的点满足 $min_x \leq x \leq max_x, min_y \leq y \leq max_y,  min_z \leq z \leq max_z$ 

构建BVH树的过程如下
- 先把所有三角形放到一个盒子里，作为根节点
- 沿着这个包围盒长/宽/高中最大的那个方向（最长轴）一分为二，并按照三角形的质心位置分到两侧，形成两个子节点
- 递归的划分下去，直到达到最大深度，或者包围盒中的三角形数量小于一个阈值

---

下面判断一个光线是否与一个包围盒相交，一个光线/射线可以参数化表示为

$$

R(t)=O+tD

$$

- $O$ 为起点，设 $O=(x_0,y_0,z_0)$ 
- $D$ 为方向，可以看作速度，设 $D=(v_x,v_y,v_z)$

我们以二维情况为例，下图黑色矩形是一个 AABB 包围盒，$A$ 和 $B$ 是其最小/最大点，一个光线（绿色）依次与 $x=min_x,y=min_y,x=max_x,y=max_y$ 相交于 $C_1\cdots C_4$ 

![[VCL/作业/img/img3/image-10.png]]

考虑光线到达 $C_i$ 的时间 $t_i$ ，以 $C_1$ 为例，我们可以把光线的运动分解为 $x$ 和 $y$ 方向的匀速直线运动，则有

$$

\begin{aligned}
&x_{OC_1}=min_x-x_0 \\
&t_1=\frac{x_{OC_1}}{v_x}=\frac{min_x-x_0}{v_x}
\end{aligned}

$$

同理可得 $t_2,t_3,t_4$ ，再结合另一条光线（橙色），可以发现以下规律：光线入射包围盒的时间是 $t_1,t_2$ 中的最大者，而出射时间是 $t_3,t_4$ 的最小者，即

$$

\begin{aligned}
t_{\text{enter}}&=\text{max}(t_1,t_2)\\
t_{\text{exit}}&=\text{min}(t_3,t_4)
\end{aligned}

$$

如果光线与包围盒相交，应满足
- $t_{\text{exit}}\geq t_{\text{enter}}$ ：确保先射入，再离开
- $t_{\text{exit}}\geq 0$ ：确保光线从正面击中（这里不要求 $t_{\text{enter}}\geq 0$ ，是因为光线从包围盒内部发出，也认为是相交）

---

下面是采样率为 1 ，最大深度为 7 的结果

![[VCL/作业/img/img3/image-9.png]]
  








