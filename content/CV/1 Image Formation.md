
# Vanishing Point/Line

在摄像机投影成像的图片中，多个平行线会汇聚于一个点，称为 **vanishing point** ，这些 vanishing point 又都处在一条水平线上，称为 **vanishing line**

![[CV/img/img1/image.png]]

对于一条直线，可以用一个点坐标加一个方向向量表示，假设这个点在世界坐标系的齐次坐标为 $X_0 = (X_0, Y_0, Z_0, 1)^T$ ，直线方向向量为 $D = (D_1, D_2, D_3)^T$ ，那这条直线的齐次坐标方程就是

$$
X_t =
\begin{pmatrix}
X_0 + tD_1 \\
Y_0 + tD_2 \\
Z_0 + tD_3 \\
1
\end{pmatrix}
\sim
\begin{pmatrix}
X_0/t + D_1 \\
Y_0/t + D_2 \\
Z_0/t + D_3 \\
1/t
\end{pmatrix}
$$

当 $t\rightarrow \infty$ 时，就到了 vanishing point ，其中最后一项是 0 ，代表这是一个无穷远处的点

$$
X_\infty =
\begin{pmatrix}
D_1 \\
D_2 \\
D_3 \\
0
\end{pmatrix}
$$

把这个点经过内参外参的投影，就能得到图片中 vanishing point 的坐标

$$v \cong P X_\infty$$

两个 vanishing point 可以确定一个 vanishing line ，在图片中，其到地面的距离（通常选取一个在地上的物体作为参照）就是相机的高度

![[CV/img/img1/image-1.png]]

# Depth of Field

在摄像机或人眼成像时，可以简化为一个凸透镜模型，可以发现只有当物体距离凸透镜处在一定距离范围内时，成像才是比较清晰的，这个范围就是 **景深 Depth of Field (DoF)** 

>可见景深实际上是一个范围，“景深大”指的是这个范围区间大，而不是具体的距离数值更大

![[CV/img/img1/image-2.png]]

DoF 受到镜头光圈的影响，光圈越大，景深越小

>改变光圈不会改变镜头焦距，但光圈变大达到的效果和使用一个长焦镜头拍摄的效果类似

![[CV/img/img1/image-3.png]]

![[CV/img/img1/image-4.png]]

- 大光圈：孔径大，进入光线多，光线锥更宽，导致景深浅，背景虚化强
- 小光圈：孔径小，进入光线少，光线锥更窄，导致景深大，前后景都清晰

# Field of View

FoV ：镜头能捕捉到的视野的角度范围

![[CV/img/img1/image-5.png]]

FoV 受到镜头焦距的影响，焦距越长，FoV 越小

![[CV/img/img1/image-6.png]]















