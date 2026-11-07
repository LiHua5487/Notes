---
date: 2025-04-02
tags:
  - CV
---

# Underfitting & Overfitting

## Underfitting on the train set

### ResNet (Residual Network)

更深的网络理论上能表达更复杂的函数，但能表达出来，不代表训练时就一定能找到这样的参数

如果一味堆叠神经网络层数会咋样？

![[CV导论/imgs/img6/image.png]]

测试和训练误差都变高，说明不是过拟合引起的
这是因为更深的模型参数更多，更难优化
反向传播时，梯度不断地被链式法则累乘更新，容易逐渐变得过小（梯度消失 gradient vanishing）或过大（梯度爆炸）
如果梯度过小，底下的层的参数很难更新
理论上来讲，只要让多出的 layer 输入输出设为一样，表现也不会比更浅的差
但实际训练不一定能找到这个恒等映射，深层网络可能出现训练误差反而增大的退化问题

#### Residual Link / Skip Link

考虑直接将一个层的输入传递到后续几层的输出
将残差块写成 $y=x+\mathcal F(x)$，其中 $x$ 是直接传过去的输入，$\mathcal F(x)$ 是中间几层学习的修正量，如果不需要改变输入，只需让 $\mathcal F(x)$ 接近 0，比要求中间几层直接学出整个恒等映射更容易

![[CV导论/imgs/img6/image-1.png|467x261]]

这样能缓解深层网络的优化困难，让较前面的层更容易获得梯度
标准的 ResNet 中，一个残差块的输出作为下一个残差块的输入，如果输入、输出尺寸或通道数不同，捷径分支还需要用投影等方式对齐，才能相加

![[CV导论/imgs/img6/image-3.png|154x290]] ![[CV导论/imgs/img6/image-2.png|184x230]]

反向传播时，梯度既能经过残差分支，也能经过直接连接，式中的 $I$ 是恒等映射的导数，表示有一条不用经过残差分支连乘的路径

$$
\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \cdot \left[ \frac{\partial \mathcal{F}(x)}{\partial x} + I \right]
$$

## Overfitting

**过拟合 overfitting**：模型在训练集上表现很好，但在没有见过的数据上表现较差，继续训练时，训练损失可能还在下降，验证效果却开始变差

![[CV导论/imgs/img6/image-4.png]]

通过引入 validation set ，可以在提升不明显或下降时提前停止，防止过拟合
那这和 test set 有啥区别？相当于模考和高考
测试集不应参与任何训练过程的决策，否则会导致测试结果存在偏差，不能准确衡量模型的泛化性能

从数据的角度，增加数据的多样性，更加充足，可以减少过拟合的发生
这有两种方式，收集更多数据，但这费时费力；还可以进行**数据增广**

### Data Augmentation

可以在不改变含义的情况下，对图片做一些简单的变换操作，比如位置方面的（旋转，裁剪等）和颜色方面的（亮度，饱和度等）

![[CV导论/imgs/img6/image-5.png]]

数据较少时，合适的增广往往尤其有帮助，数据充足时也可以用于提升鲁棒性，效果取决于任务是否允许这些变换
如果增广后 loss 不降，需要检查增广是否过强、标签是否仍然正确、模型是否难以优化，不能仅凭 loss 不降就判断为过拟合
比如区分左右手时，翻转会改变类别；裁剪可能把主体去掉，亮度调整也可能造成过曝
所以需要采样显示出来看一看

### Regularization

除了对数据操作，还可以适当限制模型，不要啥都随便学
引入**权重正则化**，可以减少过拟合发生

$$
L(W) = \frac{1}{N} \sum_{i=1}^N L_i(f(x_i, W), y_i) + \lambda R(W)
$$

- L2 regularization:

$$
R(W) = \sum_k \sum_l W_{k,l}^2 \quad \text{(Weight decay)}
$$

- L1 regularization:

$$
R(W) = \sum_k \sum_l |W_{k,l}|
$$

- Elastic net (L1 + L2):

$$
R(W) = \sum_k \sum_l \left(\beta W_{k,l}^2 + |W_{k,l}|\right)
$$

引入超参 $\lambda$ ，对正则化项做出权衡，防止其影响过大或过小
训练时需要权衡主任务损失与正则项，可以先建立不加正则的基线，再根据训练集与验证集的差距调整 $\lambda$，L2 正则与 weight decay 在普通 SGD 中关系直接，在 Adam 等自适应优化器中则需要区分

### Dropout

- 在训练过程的每一次迭代中，以$p$的概率随机丢弃一些神经元（设为0）
- 保留的神经元需要除以$1-p$，以保持一层的期望不变

$$
h' = \begin{cases} 0 & \text{概率为 } p \\ \frac{h}{1-p} & \text{其他情况} \end{cases}
$$

- 在测试时就不用随机置零了

这种随机置零可以防止对特定特征产生依赖，从而避免过拟合
并且，模型被迫学习一些更有代表性的特征，这就增强了泛化能力和鲁棒性

BN 使用每个 batch 的统计量，会给训练引入一定噪声，有时也具有正则化效果，但它不会强制输入服从高斯分布，作用也不能完全替代 dropout

早期 CNN 常在较大的 FC 层附近使用 dropout，后来一些结构用 global average pooling 减少了大 FC 层，对 dropout 的需求可能降低，具体是否使用仍取决于模型和任务

## Summary

总体上，需要让数据的多样性、模型容量和训练方法相互匹配

- 优化：BatchNorm、残差连接等
- 数据：收集更多数据、数据增广等
- 正则化：权重正则、dropout 等

可以从数据多样性、模型容量、优化和正则化几个角度综合调整，权重正则与 dropout 都不只适用于大型 FC 层；分类模型通常不直接对最终类别分数套 BN，隐藏层是否归一化则依结构而定

# Classification

之前说了二分类问题，那多分类问题呢？可以先输出 $K$ 个实数，称为 logits 或类别分数，再把它们转换为各类别的概率
把一个维度是 1，其它是 0 的 ground truth 称为 **one-hot vector**

## Softmax and CE Loss

$$
\sigma(z)_i = \frac{\exp(\beta z_i)}{\sum_{j=1}^K \exp(\beta z_j)}
$$

- $\beta$ 调节放大效应，默认为 1
越大则放大效应越大，逐渐趋于 argmax；设为 0 则变为均等分布

- $K=2$ 时，第一类的概率等于 $\operatorname{sigmoid}(\beta(z_1-z_2))$，因此可以转换为 sigmoid 二分类形式

为啥要用 softmax 而不是 argmax ？
argmax 直接输出最大分数的下标，大部分位置对输入的微小变化没有响应，在切换类别的位置又不连续，无法提供适合训练的梯度；softmax 给出可微的概率输出

![[CV导论/imgs/img6/image-6.png]]

当真实标签为 one-hot 时，可以取正确类别预测概率的负对数作为损失，下面的 $y_i$ 表示样本 $i$ 的正确类别

$$
L_i = -\log P(Y=y_i|X=x_i)
$$

如果标签本身是一个概率分布，比如经过 label smoothing，不再把所有概率都放在一个类别上，也可以使用交叉熵，或用 **KL 散度 KL divergence**衡量目标分布与预测分布的差异

$$
D_{KL}(P \parallel Q) = \sum_{x \in \mathcal{X}} P(x) \log \left( \frac{P(x)}{Q(x)} \right)
$$

- $D_{KL}(P \parallel Q) \geq 0$，当两个分布相同时为 0
- 不满足交换对称性与三角不等式

$$
\begin{aligned}
D_{KL}(P \parallel Q) &= H(P,Q) - H(P) \\
where\ H(P,Q) &= -\sum_{x \in \mathcal{X}} P(x) \log Q(x) \\
H(P) &= -\sum_{x \in \mathcal{X}} P(x) \log P(x)

\end{aligned}
$$

- $H(P,Q)$ 称为交叉熵

如果把 P 设为 ground truth，那梯度下降时，$H(P)$ 的梯度就是 0 ，只取决于交叉熵，称之为**交叉熵损失**

$$
\mathcal{L}_{CE} = H(P, Q) = -\sum_{x \in \mathcal{X}} P(x) \log Q(x)
$$

- $P(x)$ 为 ground-truth ，代表真实概率
- $Q(x)$ 为模型预测的概率

当 $P$ 是 one-hot，只有正确类别 $y$ 的权重为 1，其余为 0，因此化简为

$$
\mathcal{L}_{CE} = -\log Q(y)
$$

- 如果初始输出近似均匀，$K$ 个类别的预测概率约为 $1/K$，交叉熵约为 $\log K$，可以作为检查损失数量级的参考
- 无上界，下界为 0
