---
date: 2025-05-28
tags:
  - CV
---
# Attention

primitive：基本的单元，如 RNN CNN Attention
Attention 对于序列/非序列，关注/不关注前后信息都行

![[CV导论/imgs/img13/image.png]]

把 h 看作输入的 data vectors ，$s_i$ 与 h 进行计算，可以视为利用 query vectors “查询”二者的关系
对于更一般的情况，利用矩阵 X 和 Q 进行表示

衡量两个向量的相似度，可以用二者的夹角余弦，称 cos similarity 
参考余弦的计算，将 Q 与 X 进行点积

但有时候，Q 和 X 维度并不相同，引入 $W_k$ 把 X 转换为与 Q 相同的维度
把处理后的 $XW_k$ 称为 key 

为了防止点积结果过大，除以 $\sqrt D_Q$
再利用 softmax 归一化，得到注意力权重
$$
A=\text{Softmax}(\frac{QK^T}{\sqrt D_Q})
$$

对于输出，不一定想要 $D_x$ 维，引入 $W_v$ 处理
把处理后的 $XW_v$ 称为 value
利用上面的权重加权，就得到了最终输出
$$
Y=AV
$$
整体的计算图示如下

![[CV导论/imgs/img13/image-1.png|367x421]]

注意这里 $E_{i,j}$ 是按列进行 softmax 的

由于 X 与 Q 维度及对应的信息种类可以不同，所以称为 cross-attention
每个输出都能看见所有 X ，但只能看见一个 Q

如果 Q 也是从 X 得到的，即 $Q=XW_Q$ ，那就变成了 self-attention

![[CV导论/imgs/img13/image-2.png]]

一个句子有 N 个词，每个词通过 embedding 处理成一个 $D_{in}$ 维向量，整体作为一个输入 X ，维度为 N×D，随着单词的生成，就有不同的 X

输入只有 X ，最后输出 Y ，维度从 $D_{in}$ 变为 $D_{out}$
1D Conv 也能实现这种处理， 但 self-attention 输出的 cell 的感受野是全局的

cross-attention 对于 X 的顺序不变，对于 Q 的顺序等变
self-attention 对于输入 X 的顺序等变

但是 self-attention 不知道每个 X 在句子中的顺序
可以给每个 X 叠上一个序列顺序的信息，这个过程就是 positional encoding

## Masked Self-Attention

对于 Encoder ，可以直接用 self-attention
但对于 Decoder ，如果直接用 self-attention ，生成时就会“看到”后面的词
在一些场景，比如用语言模型生成一句话，这是有问题的
需要 causal mask ，把后面的词都遮住，不让其关注后面的词，只依赖前面的词进行输出

![[CV导论/imgs/img13/image-3.png|208x349]]

与 RNN 相比，都是依赖前面的词输出，但 RNN 仍然是串行处理的
而 masked self-attention 在训练时可以是并行的，尽管用于输出仍然是串行

## Multiheaded Self-Attention

对于同一句话，可以注意到其中的不同方面
用多个不同的 self-attention ，就成了多头注意力

![[CV导论/imgs/img13/image-4.png]]

每个 X 可以得到一堆 Y ，再经过处理合为一个输出 O ，比如 $O=YW_o$
这里，每个 Q K V 都是 $D_H$ 维
取 $D_H = D / H$， $H$ 是多头注意力中的头数，保证输入输出长度相同

计算的时间复杂度几乎不变，但是表达能力显著增强

## Summary

attention 一共经过 4 步矩阵运算

![[CV导论/imgs/img13/image-5.png|398x258]]

但是如果 N 特别大，第 2 3 步的内存与计算开销就非常大，为 $O(N^2)$
可以利用 Flash Attention ，将这两步合并，不用实际储存整个矩阵，降到 $O(N)$

三种 sequence 处理方式的对比

![[CV导论/imgs/img13/image-6.png]]

# Transformer

![[CV导论/imgs/img13/image-7.png|487x346]]

分别使用 2 层的 MLP 处理的这部分称为 FFN (Feed-Forward Network)

注意，这里 layer norm 算的是每个 token 的特征维度 channel 的平均值
比如有 $h_1 ... h_n$ ，$h_i$ 对应第 i 个 token ，维度为 D ，即 D 个 channel
$\mu_i$ 计算的是 $h_i$ 的每个元素之和的平均，即关于 channel 求均值
这与 batch norm 对于同一 channel 在所有batch与像素求均值不同

也就是说，不同 token 互动的部分只在 self-attention ，剩下部分每个 token 信息都是相互独立的

共计 6 个矩阵乘法（self-attention 4 + MLP 2）

## Vision Transformer (ViT)

如果直接处理整个图，太大了，先划成 patch
每个 patch 直接展平，用 linear 层提取一个 feature ，作为一个 token
所有 patch 的 token 传给 Transformer

![[CV导论/imgs/img13/image-8.png]]

对于每个 token ，用 positional encoding 加入这个 patch 在原图中的位置信息
由于是处理整个图片的内容，这些 token 没有先后关系，所以就不用 mask 了
对于每个 token ，都会给出一个输出，最后把这些输出经过 average pool ，就得到了整个图的特征

一些改进：

![[CV导论/imgs/img13/image-9.png]]

还可以将 layer norm 换为 RMSNorm 
$$y_i = \frac{x_i}{RMS(x)} * \gamma_i$$
- $x\ y\ \gamma$ 均为 D 维
- 通过 RMSNorm 后 x 的 L2 norm 变为一样

其中 RMS root mean square 为
$$
RMS(x) = \sqrt{\varepsilon + \frac{1}{N} \sum_{i=1}^{N} x_i^2}
$$
- 引入 $\epsilon$ 防止分母为 0


在 MLP 部分，可以将正常的 MLP 换为 SwiGLU MLP
还可以使用 Mixture of Experts (MoE)

学习 E 个权重不同的 MLP ，每个 MLP 视为一个 expert ，擅长不同方面
但实际处理时，只有 A 个 expert 参与处理，通过路由机制 rout ，根据输入选择哪些 expert 参与，这部分是需要学习的，且不用监督
最后，将这 A 个 expert 的输出进行合并

由于只选 A 个，实际计算时的参数量会比较小
但是这比 attention 更难学，因为 rout 部分只选出来 A 个，这是不可导的


由于 token 的获取方式是自由的，所以 transformer 可以处理很多模态的信息
