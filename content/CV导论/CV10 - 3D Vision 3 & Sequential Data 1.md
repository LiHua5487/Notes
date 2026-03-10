---
date: 2025-04-30
tags:
  - CV
---

把 PointNet++ 用在 MNIST 上也可以，而且表现还不错<br>
MNIST 也可视作一个点云，每个点的特征是其灰度，即 $(x,y,f=intensity)$<br>
但是 ball query 效果没有 conv 好<br>
因为 ball query 经过 PointNet 时，过的是同一个 MLP ，而后又经过 maxpool，相对位置信息就没了<br>
即 ball query 是各向同性的 isotropic ，而卷积核是各向异性 anisotropic

要把 ball query 变成各向异性，需要MLP field ，即每一个相对位置设置一个 MLP ，互不相同

hyper network 给定一个相对位置的点，预测这个位置的权重，然后再做 MLP<br>
KP conv 把一些 kernel 放到 ball 的不同位置，对于不被 kernel 涵盖的点，使用线性插值插到 kernel 里边<br>
但这些方法都比较复杂

如果处理的是 voxel 就好办了，但这太 expansive 了

# Voxel Networks

## Voxelization

采用 sparse voxel ，每个格如果被表面占据，就设成 full ，否则为 empty

![[CV导论/imgs/img10/image.png]]

处理后的 voxel 数目比点云还要少

## Sparse Conv

只在中心点是 full 的地方卷积，剩下的设为 0 

![[CV导论/imgs/img10/image-1.png]]

优点
- 比 dense conv 效率高
- 是 regular grid ，可以 indexing
- 与 2d 卷积表达能力相同，而且也对平移等变

缺点
- 离散化导致信息丢失

平均位置当作特征存在 voxel ，反映了相对位置信息<br>
尺度相比 voxel 大得多时，信息损失可忽略，而且对于大尺度，本身传感器误差就比较高，点云存储太精细就没必要<br>
点云 FPS 很慢，这么做效率高，还比 PointNet++表达力强

对于小尺度，PointNet++更常用<br>
对于大尺度，Sparse Conv 更常用

Sparse Conv
- Kernels are spatial anisotropic
- More efficient for indexing and neighbor query
- Suitable for large-scale scenes
- Limited resolutions

Point cloud networks
- High resolution
- Easier to use and can be the first choice for a quick try
- Slightly lower performance
- Slower if performing FPS and ball query

补充：
mesh 是 graph ，可以用 GNN 处理


# RNN

**sequential data** 序列化数据，有顺序
之前的神经网络，输入一个数据，输出一个结果<br>
而现在要输入一个数据，输出一个序列；或者输入一个序列，输出一个结果；还可以输入一个序列，输出一个序列<br>
比如用图片描述一句话，根据视频判断事件类型，或者总结视频

![[CV导论/imgs/img10/image-2.png]]

基本的结构是 **Recurrent Neural Network 循环神经网络**<br>
除了直接输出，还有 **internal state** ，用于向后传递之前的信息

$$
h_t = f_W(h_{t-1}, x_t)
$$

$$
y_t = f_{W_{hy}}(h_t)
$$

- 不同 t 的参数矩阵是相同的

**Vanilla RNN** <br>
计算当前时刻的隐藏状态 $h_t$

$$h_t=f(W_th_{t-1}+W_xx_t+b_h)$$

- $W_h$  $W_x$ $b_h$ 在不同时间步共享
- $f$ 为激活函数（如 tanh relu）
- $h_t$ 为时刻 t 的隐藏状态，$h_0$ 可被初始化为 0

如果当前时刻需要输出，就可以由 $h_t$ 得到

$$y_t=g(W_yh_t+b_y)$$

- $W_y$  $b_y$ 在不同时间步共享
- $g$ 为激活函数（如 softmax）
- $y_t$ 为时刻 t 的输出（如分类概率）

next token prediction <br>
训练过程中，预测下一个词时，之前的 x 用 ground-truth 而不是之前的预测

![[CV导论/imgs/img10/image-5.png]]

反向传播时，由于 W 共享，每一个 $y_i$ 对应的 $L_i$ 都会反传到之前，有 $i$ 个梯度给到 W ，W 总共受到 $1+2+...+T$ 个梯度影响

![[CV导论/imgs/img10/image-6.png]]









































