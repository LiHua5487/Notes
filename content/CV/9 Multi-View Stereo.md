
# MVS

MVS 是已知一个物体的多视角图像，以及相机参数，进行三维重建
- SFM 由于未知相机参数，还原出的三维场景一般比较稀疏，而 MVS 的结果更加密集
- MVS 一般以 SFM 为基础，通过 SFM 或其它方式给相机进行标定

MVS 的基本思想是找到图像间的密集对应关系，当一个点在多张图像中投影时，应当具有 **光度一致性 Photo-consistency** ，即表现出相似的颜色或纹理

![[CV/img/img9/image.png]]

在找到这些对应点后，可以假设 3D 点的深度（相对于某个选定的相机而言），不同的深度对应一系列光度一致性误差，选取误差最小的点即可（这是 **Plane-Sweep Stereo** 的核心思路）

![[CV/img/img9/image-1.png]]

常见的 MSV pipeline 有下面这几种

![[CV/img/img9/image-2.png]]

## Visual Hull based MVS

Visual Hull 的想法是用多个视角的物体轮廓重建出的一个近似3D形状，好比对一个物体，从不同方向用手电筒照它，在墙上投射出它的影子，根据这些影子的形状恢复出物体的三维形状（显然这种方法局限性很大，无法还原物体表面凹凸程度等细节）

如果要得到一个物体的轮廓，只需和背景比较一下，做个差分即可

![[CV/img/img9/image-3.png]]


## Depth Map based MVS

Depth Map based MVS 先根据图像还原出深度图，进而得到物体的 3D 表面信息，在还原深度图时，可以为一个图像中的每个像素在其它图像中找到最佳匹配点去计算（不同像素的最佳匹配点不一定都在同一个图像上），也可以简单的使用 two-view stereo 方法去计算

![[CV/img/img9/image-4.png]]

值得注意的是，在计算出一个深度图后，不要立刻进行正则化，而是等到所有深度图合并到一起后再对整体结果进行正则化

尽管每个深度图分别正则化看似能补充一些细节

![[CV/img/img9/image-5.png]]

但实际上，融合后往往与这个地方真实的细节差距较大，因为一个深度图看不到的地方，可能另一个深度图看到了，如果直接正则化，相当于在“臆想”这个地方的细节，这就不准确了

![[CV/img/img9/image-6.png]]

## Patch based MVS

密集的三维重建需要密集的对应关系，但如果逐像素进行，代价太大了，Patch based MVS 以一个小区域 patch 为单位，迭代的进行重建

![[CV/img/img9/image-7.png]]

一个三维空间中的 patch 具有以下属性，patch based MVS 把一堆 patch 组合成三维场景来进行重建，这可以看作是用切平面去做近似
- position ： 3D 空间中的位置
- normal ： 平面法向量方向
- extent ： 延展程度，可用半径代表

![[CV/img/img9/image-8.png]]

以 patch 为单位和处理对象有以下优点
- 由于大小适中，patch 既适用于表示物体，也适用于表示场景
- 由于 patch 是一个 3D 的东西，能直接从中提取 3D 数据，且 patch 能够携带局部特征，不会因全局正则化（如深度平滑）而丢失细节信息

![[CV/img/img9/image-9.png]]

### Photo-consistency

在 patch based MVS 中，用以下变量去描述一个 patch （记为 $p$ ）
- position ：用  $c(p)$ 表示
- normal ：用 $n(p)$ 表示
- visible images ：哪些图像中能看到这个 patch ，用一个集合 $V(p)$ 表示
- extent ：设置到合适的大小，使得在图像中的大小约为 9×9 像素

一个理想的 patch 在各个图像中看起来应该都差不多，可以用 Photo-consistency 去衡量这种一致性

考虑两个图像 $I$ 和 $J$ ，一个 patch 在二者的投影的 Photo-consistency 定义为

$$N(I, J, p) = \frac{\sum(I_{xy} - \bar{I}_{xy})(J_{xy} - \bar{J}_{xy})}{\sqrt{\sum(I_{xy} - \bar{I}_{xy})^2 \cdot \sum(J_{xy} - \bar{J}_{xy})^2}}
$$

- $I_{xy}$ 和 $J_{xy}$ 是这个 patch 在两个图像中的投影区域对应的一系列像素值
- $\bar I_{xy}$ 和 $\bar J_{xy}$ 是 patch 区域所有像素值的平均值

![[CV/img/img9/image-10.png]]

而这个 patch 在一系列图像 $V(p)={I_1,\cdots ,I_n}$ 中出现，那其对应的整体的 Photo-consistency 就是任意两个图像间的一致性的平均值

$$N(p) = \frac{\sum_{i=1}^{n} \sum_{j=i+1}^{n} N(I_i, I_j, p)}{(n+1)n/2}$$

### Refine Patch

要进行三维重建，就要优化每个 patch 的位姿，让 $N(p)$ 尽可能的大，但是由于 $V(p)$ 是一个集合，不能对 $N(p)$ 直接求导，考虑分步优化：先调整 p 的位置 $c(p)$ 和朝向 $n(p)$ ，再更新 $V(p)$ 

为了简化计算，把 $c(p)$ 的优化设置为让 patch 沿着某个选定的参考相机的深度方向移动（1 DoF），把 $n(p)$ 设置为一个单位向量（2 DoF，暂不考虑 patch 绕法向量的旋转），目标是 ${c(p),n(p)}=\text{argmax}N(p)$ 

![[CV/img/img9/image-11.png]]

之后去更新 $V(p)$ ，同样是为了让 $N(p)$ 更大，这就意味着从 $V(p)$ 中删去那些与其它图像一致性较低的图像，我们先计算出任意两个图像间的一致性，再计算一个图像与其它所有图像的一致性的和，选出最大的那个图像

![[CV/img/img9/image-12.png]]

我们把它作为参考图像（还可以作为调整 $c(p)$ 中的参考图像），设定一个阈值，把其它图像和这个参考图像比较，如果一致性低于阈值，就扔掉

![[CV/img/img9/image-13.png]]

场景中的纹理有可能偶然在两个视角之间对齐，但这不一定是合理的匹配，我们希望得到的 patch 有足够高的 Photo-consistency ，所以最后还需要检验一下得到的 patch 合不合理，可以通过以下过程判断
- 计算图像间的 Photo-consistency 并选择高一致性的视图
- 只有最后 $\lvert V(p)\rvert\geq3$ 才接受

### Feature Matching and Initialize Patch

做完图像的特征检测后，我们需要得到有哪些 patch ，并对其初始化，对于一个图像 $I_1$ 中的 patch 区域，在另一个图像 $I_2$ 中沿着极线找到对应的区域（由于 MVS 已知相机参数，可以直接计算极线），通过三角测量得到 $c(p)$ ，并把 $n(p)$ 设为 $I_1$ 的深度方向（即 patch 平面与图像平行），把 $I_1$ 和 $I_2$ 加入 $V(p)$  

![[CV/img/img9/image-14.png]]

对于其它图像，根据计算出的 patch 位姿和其它图像对应的相机参数，把 patch 投影到图像上，并计算与 $I_1$ 的一致性，大于阈值的才加入 $V(p)$ 

![[CV/img/img9/image-15.png]]

而后对这个 patch 重复前面的 refine 过程进行优化，并进行验证（更新 $V(p)$ 并检查是不是 $\lvert V(p)\rvert\geq3$ ）

一通操作之后，如果保留下了这个 patch ，就把它在各个图像中的投影区域标记为 occupied ，防止重复检测

![[CV/img/img9/image-16.png]]

### Patch expansion and Filtering

但是在特征提取与匹配时，不一定能涵盖到图像的所有部分，比如一些缺乏纹理的区域，此时需要进行 expansion 以补充这些部分对应的 patch 

由于我们缺乏这些 patch 区域的特征信息，进行不了特征匹配，那就不能直接计算其位置，需要依赖周围的 patch 来初始化，为此，我们选取一个已经得到的 patch

![[CV/img/img9/image-18.png]]

检查其在每个图像中的投影区域，周围是不是都被标记为 occupied ，如果是，说明这个 patch 的周围也有一些 patch ，其周围比较完善，啥也不用管

![[CV/img/img9/image-17.png]]

如果周围有没被标记 occupied 的区域，就说明这个 patch 周围有一些空洞，需要进行扩展来补充

![[CV/img/img9/image-19.png]]

为此，我们新建一个 patch ，记为 $q$ ，把 $n(q)$ 和 $V(q)$ 设为与 $p$ 相同，并利用 $p$ 的深度或位置信息来初始化 $c(q)$ ，比如可以设为 $p$ 的参考图像的射线（相机光心到某个像素的连线）与 $p$ 的切平面的交点

![[CV/img/img9/image-20.png]]

有了初始化后，就可以 refine 优化这个新增的 patch ，同时更新 occupied 的图像区域

![[CV/img/img9/image-21.png]]

---

扩展完后，还需要过滤掉不可靠的 patch ，只保留那些在多个照片中都看起来合理的 patch ，这可以利用 **可见一致性 visibility consistency** 来衡量，其想法是：一个 patch 应该只能被那些没有被遮挡的相机看到，比如下图 $p_1$ 挡住与 $p_2\cdots p_6$ 间存在遮挡，但其位置明显不合理，应该剔除

![[CV/img/img9/image-22.png]]

# Neural Radiance Field (NeRF)

NeRF 利用神经网络来进行 MVS 三维重建，它把场景当作一个“透明的3D空间”，里面每个点都有颜色和不透明度 $\sigma$ （这些点构成了一个场），根据输入的图像及相机参数，来预测这些点的 RGB 值和 $\sigma$ ；训练好后，当指定一个新视角，NeRF 会模拟光线穿过 3D 空间，光线上的一系列点，混合它们的颜色和密度，最终生成一张逼真的新图片

![[CV/img/img9/image-24.png]]

考虑一束光的传播过程，中间可能有一堆粒子，导致光被吸收了一部分，其亮度会衰减，假设这束光截面积为 $E$ ，粒子区域长度为 $\Delta s$ ，中间的粒子遮挡面积为 $A$ ，体密度为 $\rho$ ，则被遮挡的总面积为

$$S=(\rho E\Delta s)A$$

单位面积的遮挡率为

$$\frac{S}{E}=\rho \Delta s A$$

假设光线初始强度为 $I$ ，受到遮挡导致的亮度衰减系数为 $k$ ，则单位长度的衰减程度为

$$\frac{\Delta I}{\Delta s}=-(k\rho A)I$$

把光沿着传播方向的亮度视为一个关于传播距离 $s$ 的函数 $I(s)$ ，记 $\tau = k\rho A$ ，它代表了粒子的遮挡程度，由于不同位置的粒子可能也不同，所以把 $\tau$ 也视为关于 $s$ 的函数 $\tau(s)$ ，则

$$\frac{dI(s)}{ds}=-\tau(s)I(s)$$

求解可得

$$I(s) = I_0 e^{-\int_{0}^{s} \tau(t) dt}$$

将其中的幂次定义为透明度 $T$ 

$$T(s) = \exp\left(-\int_{0}^{s} \tau(t) dt\right)$$

也可以进一步定义不透明度

$$\alpha = 1 - T(s) = 1 - \exp\left(-\int_{0}^{s} \tau(t) dt\right)$$

把光线沿着传播方向划分为一堆长度区域 0 的小区段（对应到三维重建场景中的一个点），则每个小区段上 $\tau$ 可以认为是不变的，则不透明度简化为

$$\alpha_i=1-e^{\tau_i\delta_i}$$

一条光线会经过若干个点最终到达成像平面，假设一个点颜色为 $c_i$ ，考虑到不透明度后，其实际颜色就是 $\alpha_ic_i$ ，其在传播过程中会受到前面的一堆点的遮挡，那么它对最终颜色的贡献就是

$$\left(\prod_{j=1}^{i-1} (1-\alpha_j)\right)\alpha_ic_i$$

则光线上的一系列点叠加后的结果就是

$$\hat{c}(r) = \sum_{i=1}^{N} \left(\prod_{j=1}^{i-1} (1-\alpha_j)\right)\alpha_ic_i
$$

在预测出三维场景后，采样一些光束，重投影到图像上，比较其像素颜色与 gt 像素颜色的差距，以此为误差进行训练

![[CV/img/img9/image-23.png]]

但是 NeRF 要整这么多点，太慢了，Gaussian Splatting 把点带来的颜色影响改为一个高斯分布，类似于一个颜色泡，用一堆泡泡去还原场景

![[CV/img/img9/image-25.png]]
























