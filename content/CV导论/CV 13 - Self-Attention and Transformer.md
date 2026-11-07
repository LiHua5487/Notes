---
date: 2025-05-28
tags:
  - CV
---

# Attention

**注意力 attention**也是一种基本计算单元 primitive，可以处理有顺序的序列，也可以处理图像块或其他向量集合，核心是根据当前需要，从一组信息中加权提取有用内容

![[CV导论/imgs/img13/image.png]]

沿用上一讲，$h$ 表示输入信息，$s_i$ 表示当前查询状态，可以把 $s_i$ 看作 **query（查询）**：现在需要什么信息？把输入变成 **key（键）**后，用来判断相关程度；再用这些相关程度，对 **value（值）**进行加权求和

对于一般情况，用矩阵 $X$ 存放待查询的输入，每行一个向量；用 $Q$ 存放查询，每行一个 query，query 的数量可以与输入数量不同

两个向量的点积可以衡量匹配程度，除以各自模长后才是余弦相似度 cosine similarity，下面使用的是缩放点积注意力，而不是归一化后的余弦相似度

但有时候，Q 和 X 维度并不相同，引入 $W_k$ 把 X 转换为与 Q 相同的维度
把 $K=XW_k$ 称为 key，让每个 key 与 query 的维数相同，记为 $d_k$

为了避免维数较大时点积的尺度随之增大，除以 $\sqrt{d_k}$
再利用 softmax 归一化，得到注意力权重

$$
A=\operatorname{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)
$$

>与初始化时控制信号尺度类似，如果各分量近似独立、均值为 0、方差为 1，那么 $d_k$ 项乘积之和的方差约为 $d_k$，除以 $\sqrt{d_k}$ 后尺度更稳定，能避免 softmax 过早变得过于尖锐

对于输出，不一定想要 $D_x$ 维，引入 $W_v$ 处理
把处理后的 $XW_v$ 称为 value
利用上面的权重加权，就得到了最终输出

$$
Y=AV
$$

整体的计算图示如下

![[CV导论/imgs/img13/image-1.png|367x421]]

图中每列固定一个 query、每行对应一个 key，所以沿列内各行做 softmax；上面的 $QK^T$ 写法则每行固定一个 query，应沿每行的 key 维度归一化，两种画法只是矩阵摆放方向不同，核心都是“为同一个 query 分配总和为 1 的权重”

当 $Q$ 来自一组表示，$K,V$ 来自另一组表示时，称为 **cross-attention（交叉注意力）**，例如用 decoder 的状态查询 encoder 的输出，两组表示的来源不同是关键，并不要求它们维数或模态一定不同
每个输出都能看见所有 X ，但只能看见一个 Q

如果 Q 也是从 X 得到的，即 $Q=XW_Q$ ，那就变成了 self-attention

![[CV导论/imgs/img13/image-2.png]]

一个句子有 $N$ 个 token，每个经过 embedding 得到 $D_{in}$ 维向量，叠成 $X\in\mathbb R^{N\times D_{in}}$，self-attention 中分别计算 $Q=XW_Q$、$K=XW_K$、$V=XW_V$，让各 token 根据当前内容相互交换信息

输入只有 X ，最后输出 Y ，维度从 $D_{in}$ 变为 $D_{out}$
1D Conv 也能实现这种处理， 但 self-attention 输出的 cell 的感受野是全局的

暂不考虑位置编码与固定顺序的 mask，cross-attention 中将 $X$ 的各行同步重排，会同时重排 key 与 value，最终加权和不变；重排 query，则输出按同样顺序重排

self-attention 中将输入重排，输出也相应重排，即排列等变，因此它本身不知道 token 原来处于第几个位置，可以给每个 token 叠上位置相关的信息，称为**位置编码 positional encoding**

## Masked Self-Attention

处理完整输入的 encoder 通常可以让各位置互相关注
但在自回归 decoder 的训练中，完整目标序列通常已经给定，如果不遮挡，当前位置就可能直接利用未来 token，相当于提前看到了答案
在一些场景，比如用语言模型生成一句话，这是有问题的
需要 **causal mask（因果掩码）**，把未来位置的注意力分数设为 $-\infty$，softmax 后这些位置的权重就是 0，让当前位置只利用允许看到的前缀，实际生成时未来 token 本来就还不存在

![[CV导论/imgs/img13/image-3.png|208x349]]

与 RNN 相比，都是依赖前面的词输出，但 RNN 仍然是串行处理的
而 masked self-attention 在训练时可以是并行的，尽管用于输出仍然是串行

## Multiheaded Self-Attention

对于同一句话，可以注意到其中的不同方面
用多个不同的 self-attention ，就成了多头注意力

![[CV导论/imgs/img13/image-4.png]]

每个头使用各自的 $W_Q,W_K,W_V$，得到一组输出，将各个头的输出沿特征维拼接 concat，再乘 $W_O$ 得到最终输出，即 $O=\operatorname{Concat}(Y^{(1)},\ldots,Y^{(H)})W_O$
这里，每个 Q K V 都是 $D_H$ 维
取 $D_H = D / H$， $H$ 是多头注意力中的头数，保证输入输出长度相同

在总特征宽度 $D$ 固定时，每个头的宽度约为 $D/H$，主要计算量仍在同一个量级，多头让模型有机会学习不同的关注方式，但不保证每个头都自动对应某种明确语义

## Summary

图中把 key、value 的投影、相关性计算和加权聚合画成 4 个矩阵乘法；完整的多头 self-attention 通常还包含 query 投影和最终输出投影，具体计数要看把哪些步骤算进模块

![[CV导论/imgs/img13/image-5.png|398x258]]

但是如果 N 特别大，第 2 3 步的内存与计算开销就非常大，为 $O(N^2)$
可以利用 **FlashAttention**按块计算、融合操作，避免把完整的 $N\times N$ 注意力矩阵写入显存；在特征维固定时，额外内存可随 $N$ 线性增长，但精确稠密注意力的算术计算量仍然是二次的，主要通过减少显存读写来加速

三种 sequence 处理方式的对比

![[CV导论/imgs/img13/image-6.png]]

# Transformer

![[CV导论/imgs/img13/image-7.png|487x346]]

Transformer block 将 attention、残差连接、归一化和逐 token 的 MLP 组合起来，其中通常包含两次线性变换及中间激活的 MLP 称为 **前馈网络 FFN (feed-forward network)**，不同 token 共用 FFN 参数

注意，这里 layer norm 算的是每个 token 的特征维度 channel 的平均值
比如有 $h_1 ... h_n$ ，$h_i$ 对应第 i 个 token ，维度为 D ，即 D 个 channel
$\mu_i$ 计算的是 $h_i$ 的每个元素之和的平均，即关于 channel 求均值
这与 batch norm 对于同一 channel 在所有batch与像素求均值不同

也就是说，不同 token 互动的部分只在 self-attention ，剩下部分每个 token 信息都是相互独立的

因此理解计算过程时，要把 attention 内的投影、两次注意力矩阵乘法、输出投影与 FFN 分开看，采用常见完整定义时，多头 self-attention 可按 6 次主要矩阵乘法计，FFN 再增加 2 次

## Vision Transformer (ViT)

如果直接处理整个图，太大了，先划成 patch
每个 patch 直接展平，用 linear 层提取一个 feature ，作为一个 token
所有 patch token 传给 Transformer，patch 相当于把一小块图像当成一个“词”，避免直接把每个像素都当作 token 带来的过长序列

![[CV导论/imgs/img13/image-8.png]]

对于每个 token ，用 positional encoding 加入这个 patch 在原图中的位置信息
整图分类时，各 patch 都是已经给定的输入，不需要像自回归生成那样遮住未来位置，因此普通 ViT 可以让它们相互关注
每个 token 都会得到输出，可以对它们做 average pooling 得到整图特征，也可以像原始 ViT 那样加入可学习的分类 token `[CLS]`，用其最终输出进行分类

一些改进：

![[CV导论/imgs/img13/image-9.png]]

还可以将 layer norm 换为 RMSNorm

$$
y_i = \frac{x_i}{RMS(x)} * \gamma_i
$$

- $x\ y\ \gamma$ 均为 D 维
- RMSNorm 用均方根缩放输入，不减去均值；忽略 $\epsilon$ 和后续可学习缩放时，归一化向量的 L2 范数为 $\sqrt D$，加入不同的 $\gamma_i$ 后不再保证最终范数完全相同

其中 RMS root mean square 为

$$
RMS(x) = \sqrt{\varepsilon + \frac{1}{D} \sum_{i=1}^{D} x_i^2}
$$

- 引入 $\epsilon$ 防止分母为 0

在 FFN 部分，可以换用 **SwiGLU**，用一个经过 SiLU 激活的分支对另一个分支逐元素门控，再投影回原来的宽度
还可以使用 Mixture of Experts (MoE)

学习 E 个权重不同的 MLP ，每个 MLP 视为一个 expert ，擅长不同方面
但对于每个 token，只选择 $A$ 个 expert 参与计算，通常 $A<E$，通过**路由 router**根据输入给各 expert 打分，选取其中的 top-$A$，路由器可以随任务损失训练，通常不需要人工标注应该选择哪个 expert
最后，将这 A 个 expert 的输出进行合并

由于只选 A 个，实际计算时的参数量会比较小
top-$A$ 的离散选择本身不可导，但选中专家的路由权重和输出仍可以传播梯度，实际还需要处理专家负载不均、部分专家很少被选中等训练问题

由于 token 的获取方式是自由的，所以 transformer 可以处理很多模态的信息
