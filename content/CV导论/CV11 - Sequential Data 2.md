---
date: 2025-05-14
tags:
  - CV
---
# RNN

可以堆叠多个 RNN ，有多次矩阵乘法，增强表达能力

![[CV导论/imgs/img11/image-5.png]]

也可以前文向前传一个 h ，后文往回传一个 h ，上下文信息就都考虑到了，但这就不能用于输出了（输出时没有下文信息），只能用于处理整段文字

## Truncation

训练时，为了防止训练过长，采用**截断 Truncation**<br>
将输入序列分割成等长的片段，每次只输入这些子序列到 RNN 中

![[CV导论/imgs/img11/image.png]]

其中的 $h_0$ 如何获取？

前向截断 Truncated Sequence Processing<br>
$h_0$ 仍然用 0 ，保证效率，但这使得这个单元只能见到定长的序列，更长程的关系可能学不到

![[CV导论/imgs/img11/image-1.png]]

全序列的前向传播依然会执行，但是反向传播只在固定的时间步长度上执行<br>
这样对于这个单元，$h_0$ 就是前面的隐藏输出，具有之前的语义信息 context <br>
但如果这么做，前面参数更新后，后面的单元的 $h_0$ 就变了，训练时需要把整个网络与输入序列都过一遍

## Beam Search

在输出时，有以下策略

Greedy Sample ：每次取 Softmax 后概率最高的词
- 对于特定的开头输入，只能产生固定的输出

Weighted Sample ：根据概率分布产生输出，不只考虑最高的
- 可以有多样的输出
- 但是有小概率可能选一个不合理的，偏差越来越大

输出每个词时，考虑前面 T 个词，其概率为

$$
\begin{align}
P(y|x) &= P(y_1|x) P(y_2|y_1, x) P(y_3|y_1, y_2, x) \cdots P(y_T|y_1, \dots, y_{T-1}, x) \\
&= \prod_{t=1}^T P(y_t | y_1, \dots, y_{t-1}, x)
\end{align}
$$

考虑所有的组合，使这个概率最大化，但这开销很大

采用 beam search，每次只保留 top k 个概率，k 称 beam size<br>
对于第一个输入，有 k 个可能的输出，每个输出又对应 k 个下一个词，对于这 $k^2$ 个词，再取前 k 个<br>
虽然不一定保证最优，但效率更高

## Embedding Layer

有时候，输入的词可能有很多种，如果直接用 one-hot vector ，维数会很大<br>
可以先转化为一个向量，即词嵌入 embedding

![[CV导论/imgs/img11/image-2.png]]

这部分一般是预训练好的

## Gradient Flow

![[CV导论/imgs/img11/image-3.png]]

![[CV导论/imgs/img11/image-4.png]]

将 $\frac{\partial h_t}{\partial h_{t-1}}$ 其带入，得

$$\frac{\partial L_T}{\partial W} = \frac{\partial L_T}{\partial h_T} \left( \prod_{t=2}^T tanh'(W_{hh} h_{t-1} + W_{xh} x_t) \right) W_{hh}^{T-1} \frac{\partial h_1}{\partial W}
$$

其中有 $tanh'$ 的连乘，而 $tanh' < 1$，会导致梯度消失，这进一步限制了长程学习能力，使得实际上考虑的信息小于 sequence length，最终导致生成的内容前言不搭后语

如果去掉 tanh ，连乘仍然可能导致梯度消失/爆炸，解决不了问题，对于梯度爆炸，可以设一个上限，对于梯度消失，不能简单的直接缩放，要调整结构

## Summary

RNN 优点
- 能处理任意长度的输入序列
- 理论感受野是全文（但实际输出时只考虑了前面一部分的词）
- 模型大小不会随着输入长度增加而增大
- 权重矩阵共享，对于输入的处理

缺点
- 需要不断循环，速度慢
- 长程关系的学习能力差

# LSTM (Long Short Term Memory)

$h_t$ 只考虑前面一定长度的词，可以视作短程记忆<br>
考虑引入长程记忆 $c_t$ ，称为 cell state<br>
$h_t$ 与 $c_t$ 等长，对于 cell state ，可以读写与清空

![[CV导论/imgs/img11/image-6.png]]

对于前面的 hidden state $h_{t-1}$ 与当前输入 $x_t$ ，先计算得到 4 个值

![[CV导论/imgs/img11/image-7.png]]

得到的的 $i\ f\ o\ g$ 称为 gate ，长度与 h 和 c 相同，据此进一步计算 $c_t$ 与 $h_t$
- $i$ ：Input gate ，控制当前的信息要不要写入 cell
- $f$ ：Forget gate ，控制要不要遗忘之前的 cell
- $o$ ：Output gate ，控制 cell 的内容怎么影响到输出
- $g$ ： Gate gate ，控制当前信息写入 cell 的程度
其中对于 $c_t$ 的更新方式就属于 additive interaction

这个单元的计算图如下，最上面 $c_t$ 的传播就类似于 skip link<br>
极端情况 $f = 1,\ i = 0$ ，那 $c_t$ 永远不变

![[CV导论/imgs/img11/image-8.png]]

# Sequence to Sequence

给一段序列，输出一个序列，比如翻译问题，与之前的文本生成不同（可以只给一个开始的词）

可以使用 RNN 实现，引入两个特殊 token `[start]` `[stop]`

![[CV导论/imgs/img11/image-9.png]]

通过 encoder 加密上下文信息，再利用 decoder 生成输出，但 encoder 压缩到一个特征向量，可能导致信息丢失

补充：
Fusion 信息聚合
- 直接加
- 逐元素乘 point/element-wise multiply
- concat
ablation study 单变量原则对比实验<br>
multi-modal 多模态

## Attention

在 segmentation 中，bottleneck 的问题可以通过 skip  link 解决<br>
但对于 Seq2Seq ，显然不能直接把句子中的一段词直接传给后面，需要整句处理后传给后面，生成每个词都能看到整句信息

![[CV导论/imgs/img11/image-10.png]]

每个 hidden state 先经过一个 attention 函数，再用 softmax 归一化，作为注意力权重，代表注意力的分配，最后将 hidden state 加权求和作为上下文信息<br>
这部分是自己学习的，且无需监督

生成下一个词时，把前一个 hidden state 再经过 attention 获取到另一个上下文向量

![[CV导论/imgs/img11/image-11.png]]

由于每个上下文向量都不同，可以避免 bottleneck ，同时每个上下文向量关注了原句的不同侧重的信息








