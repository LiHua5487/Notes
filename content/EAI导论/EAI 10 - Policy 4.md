
想要学习连续的动作，一种方法是**离散化 discretization** ，转化成分类问题，但是原本连续空间里的相似的动作，离散化后会变为完全不同的分类标签，相差可能很大，影响模型的学习效果（比如泛化能力差、误差容易累积）

高斯策略假设动作是一个高斯分布，但很多情况下动作分布并不是单峰的；GMM 用一堆高斯分布去拟合，但是难以学习，而且还是太生硬了（比如需要提前设定高斯分布的个数、对协方差敏感）

# Diffusion Policy

考虑使用扩散模型去学习动作分布
- 给动作序列加噪 / 去噪
- 将 observation 作为条件信息

## Flow Matching

Diffusion 的方法，是先给图像一步步加噪，在去噪时让模型预测“一个带噪声的图像里有多少噪声”，而 Flow-Matching 是学习一个速度场，输入当前图像（位置），判断下一步要往哪移动（速度）

![[EAI导论/imgs/plc4/image.png]]

训练过程中，随机采样一个噪声 $z$ 和图像 $x$ ，把它们直接连起来（不需要从 $x$ 一步步加噪），在连线上插值得到插值点 $x_t$ （总时间为 1）

$$
x_t = (1-t)x + tz
$$

并定义速度

$$
v = z - x
$$

模型需要学习：在 $t$ 时刻，看到图片 $x_t$ ，需要往哪移动，即 $f_\theta(x_t,t)$ ，可得损失函数

$$
L = \lVert f_\theta (x_t,t) - v \rVert_2^2
$$

>虽然是随机连线，而且可能发生多次采样中，同一个 $x_t$ 有不同的速度，但数学证明这种方式是可行的

---

生成时，先采样一个噪声，再根据速度逐步往回走（设置总步数 $T$ ，则每一步时间为 $\frac{1}{T}$ ）

![[EAI导论/imgs/plc4/image-1.png]]

## Conditional Flow Matching

加入条件信息 $y$ 

![[EAI导论/imgs/plc4/image-2.png]]

## Classifier-Free Guidance (CFG)

想要控制条件 $c$ 的影响程度，传统的方法是额外搞一个分类器 $p_\phi(c\mid x_t)$ ，用来判断当前图像 $x_t$ 有多像条件 $c$ ，在采样时让图像朝着更靠近条件的方向走，即 $\nabla_{x_t} \log p_\phi(c\mid x_i)$ 

而 CFG 不使用额外分类器，而是让模型在训练时同时学习有条件和无条件的生成，把一些数据的条件标签随机设为空，这样模型就会学到无条件速度 $v^\emptyset$ 和有条件速度 $v^y$ ，最终学到的速度是

$$
v^{cfg} = v^y + w(v^y - v^\emptyset)
$$

- $v^y - v^\emptyset$ 表示有条件与无条件相差的方向
- 在 $v^y$ 的基础上叠加 $w(v^y - v^\emptyset)$ ，相当于增强条件的影响

![[EAI导论/imgs/plc4/image-3.png]]

## Noise Schedules

前面提到多次采样中，可能发生同一个 $x_t$ 有不同的速度的情况
- 当 $t$ 更靠近 0/1 ，即更靠近图像/噪声时，$x_t$ 不容易发生重合（因为端点处的采样很难发生重复）
- 当 $t$ 处在路径中间时，就容易发生重合，模型需要更多的学习这种情况

![[EAI导论/imgs/plc4/image-5.png]]

考虑将 $t$ 的采样改为使用 sigmoid 函数，增加中间部分的比例


![[EAI导论/imgs/plc4/image-4.png]]

## Generalizing Flow Matching

连接图像与噪声时，不一定非得用直线，可以扩展成任意曲线

![[EAI导论/imgs/plc4/image-6.png]]

当 $a(t) = \sqrt \alpha, b(t) = \sqrt{1-\alpha}$ 时，就变成了 Diffusion 模型
