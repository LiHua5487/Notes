---
date: 2025-04-02
tags:
  - CV
---
# Underfitting & Overfitting

## Underfitting on the train set

### ResNet (Residual Network)

如果一味堆叠神经网络层数会咋样？

![[imgs/img6/image.png]]

测试和训练误差都变高，说明不是过拟合引起的
这是因为更深的模型参数更多，更难优化
反向传播时，梯度不断地被链式法则累乘更新，容易逐渐变得过小（梯度消失 gradient vanishing）或过大（梯度爆炸）
如果梯度过小，底下的层的参数很难更新
理论上来讲，只要让多出的 layer 输入输出设为一样，表现也不会比更浅的的差
但实际上，这是训练不出来的

#### Residual Link / Skip Link

考虑直接将一个层的输入传递到后续几层的输出
这样后面的层只需要学到输出为 0 的就可以了，相当于数学上的残差

![[imgs/img6/image-1.png|467x261]]

这样可以避免更多层数导致表现变差，同时底层的参数能更好的更新
标准的 ResNet 中，一个残差块的输出直接作为下一个残差块的输入

![[imgs/img6/image-3.png|154x290]] ![[imgs/img6/image-2.png|184x230]]

反向传播时，分成两股传播
$$
\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \cdot \left[ \frac{\partial \mathcal{F}(x)}{\partial x} + I \right]
$$

## Overfitting

一直训可能导致过拟合

![[imgs/img6/image-4.png]]

通过引入 validation set ，可以在提升不明显或下降时提前停止，防止过拟合
那这和 test set 有啥区别？相当于模考和高考
测试集不应参与任何训练过程的决策，否则会导致测试结果存在偏差，不能准确衡量模型的泛化性能

从数据的角度，增加数据的多样性，更加充足，可以减少过拟合的发生
这有两种方式，收集更多数据，但这费时费力；还可以进行**数据增广**

### Data Augmentation 

可以在不改变含义的情况下，对图片做一些简单的变换操作
比如位置方面的（旋转，裁剪等）和颜色方面的（亮度，饱和度等）

![[imgs/img6/image-5.png]]

在数据较少时，这么做的效果比较明显，但数据本身就很多，那就不是很明显
如果这么做了但是 loss 不降，说明模型有问题（可能过拟合）
或者增广操作有问题，比如左手翻转变右手，把主体给裁剪没了，亮度过曝了
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
R(W) = \sum_k \sum_l \beta W_{k,l}^2 + |W_{k,l}|
$$
引入超参 $\lambda$ ，对正则化项做出权衡，防止其影响过大或过小
实际过程中，可以先不加 $\lambda$ ，在 main loss 不降后，再适当调整 $\lambda$ 

### Dropout

- 在训练过程的每一次迭代中，以$p$的概率随机丢弃一些神经元（设为0）
- 保留的神经元需要除以$1-p$，以保持一层的期望不变
$$ h' = \begin{cases} 0 & \text{概率为 } p \\ \frac{h}{1-p} & \text{其他情况} \end{cases} $$
- 在测试时就不用随机置零了

这种随机置零可以防止对特定特征产生依赖，从而避免过拟合
并且，模型被迫学习一些更有代表性的特征，这就增强了泛化能力和鲁棒性

但是有了 batch norm 后，dropout 的应用就变少了，因为 batch norm 的好处涵盖了 dropout 的好处
那 batch norm 的好处是咋来的？

把限制模型的表达能力称为 regularization
batch norm 限制了输入数据必须服从某个高斯分布，从而减少过拟合
而这同时也利于优化，进而减少 underfitting

## Summary

Principle
  - to balance the data variability and the model capacity

Techniques
-  **BatchNorm** 
- Data
	- **Data augmentation** 
- Model
	- Regularization
	- Dropout
  
加粗的一般情况下都能用，没加粗的一般只用于大型 FC layer
但是 batch norm 在最后的 FC 就不用了，因为要最后分类，不能乱改输入分布

# Classification

之前说了二分类问题，那多分类问题呢？
输出设为一个向量，每个维度代表某类的概率
把一个维度是 1，其它是 0 的 ground truth 称为 **one-hot vector** 

## Softmax and CE Loss

$$
\sigma(z)_i = \frac{\exp(\beta z_i)}{\sum_{j=1}^K \exp(\beta z_j)}
$$
- $\beta$ 调节放大效应，默认为 1
	越大则放大效应越大，逐渐趋于 argmax；设为 0 则变为均等分布
- $K=2$ 时即为 sigmoid 函数

为啥要用 softmax 而不是 argmax ？
因为 argmax 只有在值的交界处有梯度，不利于反向传播

![[imgs/img6/image-6.png]]

当 ground truth 为 one-hot 时，可以采用 NLL loss
$$L_i = -\log P(Y=y_i|X=x_i)$$
当 ground truth 不是 one-hot 的，意味着有不确定性，或经 label smoothing，可以采用 **KL-divergence** 衡量差异
$$
D_{KL}(P \parallel Q) = \sum_{x \in \mathcal{X}} P(x) \log \left( \frac{P(x)}{Q(x)} \right)
$$
- $D_{KL}(P \parallel Q) \geq 0$  当且仅当 $P=Q$ 时取等
- 不满足交换对称性与三角不等式

$$
\begin{align}
D_{KL}(P \parallel Q) &= H(P,Q) - H(P) \\
where\ H(P,Q) &= -\sum_{x \in \mathcal{X}} P(x) \log Q(x) \\
H(P) &= -\sum_{x \in \mathcal{X}} P(x) \log P(x)

\end{align}
$$
- $H(P,Q)$ 称为交叉熵

如果把 P 设为 ground truth，那梯度下降时，$H(P)$ 的梯度就是 0 ，只取决于交叉熵，称之为**交叉熵损失**
$$
\mathcal{L}_{CE} = H(P, Q) = -\sum_{x \in \mathcal{X}} P(x) \log Q(x)
$$
- $P(x)$ 为 ground-truth ，代表真实概率
- $Q(x)$ 为模型预测的概率

当 P 是 one-hot 时，可以进一步化简为
$$\mathcal{L}_{CE} = -\log Q(x)$$
- 随机初始化时，$Q(x) = \frac{1}{N}$，交叉熵损失为 $\log N$
- 无上界，下界为 0






















