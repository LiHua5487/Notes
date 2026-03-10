---
date: 2025-06-04
tags:
  - CV
---

# Generative Model

生成东西，比如一个图片时，每个像素的每个可能的取值都有对应概率，并从中选取，所以可以建模成以下过程
- 学习训练数据的概率分布，即从 $p_{data}$ 学习 $p_{model}$
- 生成时，根据 $p_{model}$ 进行采样

这里的 $p$ 代表的是概率密度，所以可以大于1

![[CV导论/imgs/img14/image.png|466x275]]

判别式模型学习条件概率分布，生成式模型学习联合概率分布

生成式模型分类
- explicit density ：显示定义分布
	- tractable ：得到解析解
	- approximate ：近似求解，如 VAE Diffusion
- implicit density ：不定义分布，如 GAN

一个简单的 explicit density model 表达式如下
$$p(x) = p(x_1, x_2, \dots, x_n)$$
- $p(x)$ 表示图片 $x$ 的概率，每个 $x_i$ 表示图像的一个像素
利用 chain rule ，可以表示成这样
$$p(x) = \prod_{i=1}^n p(x_i | x_1, \dots, x_{i-1})$$
指定 $x_i$ 的位置顺序，类似于RNN逐个 $x_i$ 去生成
这是 auto regressive  的，但是效率太低了

## Variational Autoencoders (VAE)

本质上是一个概率性的 autoencoder ，而且是 self-supervise 的，不用 label

对于决定性的 autoencoder，对于相同的输入，输出是固定的

![[CV导论/imgs/img14/image-1.png|327x211]]

但这本身并不能生成东西，因为既没有建模概率分布，也不能从中采样

想要生成，就要把 z 建模成概率分布，从输入的概率分布学习
生成时，先采样 z ，再通过 decoder 采样进行输出
而训练时，给出训练数据（真实的图片），经过 encoder 得到 z ，再经过 decoder 输出

可以把 z 看作对于图片事物的概念的理解
生成时，先选出一种概念，据此生成图片
而训练时，给出一些真实图片，学习这个概念，再据此重建

但我们想学的是符合训练数据的分布，所以要计算关于训练数据的分布 $p(x|z)$

![[CV导论/imgs/img14/image-2.png|348x198]]

假设 z 的分布 $p(z)$ 是标准正态分布
再假设采样过程 $p(x|z)$ 也是一个高斯分布，只用去预测 $\mu$ 和 $\sigma$
据此可以得到 x 的分布
$$p_{\theta}(x) = \int p(z) p_{\theta}(x|z) dz$$
但积分很难求，即便使用蒙特卡洛近似，梯度噪声非常大

考虑另一种表示方式
$$p_{\theta}(x) = \frac{p_{\theta}(x, z)}{p_{\theta}(z|x)} = \frac{p_{\theta}(z) p_{\theta}(x|z)}{p_{\theta}(z|x)}$$
要求 $p_{\theta}(z|x)$ ，但不能直接求，不过这部分相当于 encoder 的过程，考虑建模成高斯分布，用 $q_{\phi}(z|x)$ 去近似，也对应一组 $\mu$ 和 $\sigma$

整体 VAE 的结构如下

![[CV导论/imgs/img14/image-3.png]]

采用 MLE ，VAE 的训练目标是最大化边缘似然 marginal likelihood
即模型认为数据 x 出现的概率，最大化这个概率分布，就是让学到的分布尽可能接近输入数据的概率分布

![[CV导论/imgs/img14/image-4.png]]

但是最后一项 $p_{\theta}(z|x)$ 与 $q_{\phi}(z|x)$ 的 $D_{KL}$ 算不了
使用变分近似，把最后一项扔了，由于这一项非负，所以扔掉后是一个下界
而 $\mathbb{E}_{z} \left[ \log p_{\theta}(x^{(i)}|z) \right]$ 这部分也是很难计算的，用蒙特卡洛近似，去掉 $\mathbb{E}_{z}$
$$\mathcal{L}_{\text{ELBO}} = \log p_{\theta}(x^{(i)}|z) - D_{KL}(q_{\phi}(z|x^{(i)}) \| p(z))$$
对于 $D_{KL}(q_{\phi}(z|x^{(i)}) \| p(z))$ ，由于假设为高斯分布，就是 $D_{KL}(\mathcal{N}(\mu_{z|x}, \sigma_{z|x}) \| \mathcal{N}(0, I))$ ，这一部分是有解析形式的

对于从 x 采样 z 这部分，原本是不可导的，采用重参数化技巧变为可导
$$
z = \mu_{z|x} + \epsilon \cdot \sigma_{z|x}, \quad \epsilon \sim \mathcal{N}(0, I)
$$
从采样 z ，变为根据高斯分布随机采样 $\epsilon$ 

而训练时，decoder 部分并没有进行实际采样生成 $\hat{x}$ ，只是计算 $p_\theta(x|z)$ ，所以这部分是可导的
于是整体就变成了端到端可导的

在实际实现时，可以取 $\sigma_{z|x} = 1$ ，那 $\log p_{\theta}(x^{(i)}|z)$ 这部分就变成了 MSE 损失
$$
\| x^{(i)} - \mu_{x|z} \|^2
$$

训练时要最大化 $\mathcal{L}_{\text{ELBO}}$ ，第一项 $\log p_{\theta}(x^{(i)}|z)$ 要尽量大，要求通过 $z$ 尽可能重构出真实分布， $z$ 和 $x$ 相关性增大；第二项 $D_{KL}(q_{\phi}(z|x^{(i)}) \| p(z))$ 要尽量小，
要求 $q_{\phi}(z|x^{(i)})$ 趋近于标准正态分布 $\mathcal{N}(0, I)$ ，$z$ 和 $x$ 相关性减小

所以这两项是矛盾的，不能完全训练
不过至少可以通过这种计算方式给出 log likelihood 的下界

补充
- 泛函：函数映射到实数
- 变分：泛函对应的”微分“

## Generative Adversarial Networks (GAN)

回归问题本身，我们希望能够从训练数据的真实分布（一个复杂的高维分布）中采样，然而，有以下难点
- 真实分布复杂且不可直接建模：训练数据的分布是复杂的高维分布，我们无法直接对其进行解析或者直接从中采样
- 高维采样困难： 直接去学高维复杂分布，或者直接生成采样点非常困难

VAE 只是简单的建模为高斯分布，效果并不理想

![[CV导论/imgs/img14/image-6.png|334x202]]

考虑先用简单的分布作为输入（如随机噪声），这是容易进行采样的
通过一个生成模型 $G(z)$ ，学会如何将简单分布中的样本 $z$ 转化为复杂的高维目标分布中的样本

直观上，这个生成网络就像是一种转换器，能通过学习训练数据特征，把随机噪声 $z$ 映射成看起来“像真实样本”的数据

那要用什么方式去评价呢？

如果直接按照与输入数据图片的差异作为 loss ，那就没有生成新图的能力了
如果用 L2 Loss ，那生成时每个 pixel 会自带 gauss 噪声，不清晰

把 loss 这部分换成一个判别式模型，称 discriminator ，评价这个图是真的假的
但 discriminator 也需要训练，在训练时，把 generator 生成的图片认为是假的，从数据集选取的图片认为是真的，辨别差异
这样，generator 就需要生成尽可能真的图片通过 discriminator 的判断

![[CV导论/imgs/img14/image-5.png]]

训练时，采用 minimax 的方式 
$$
\min_{\theta_g} \max_{\theta_d} \left[ \mathbb{E}_{x \sim p_{\text{data}}} \log D_{\theta_d}(x) + \mathbb{E}_{z \sim p(z)} \log(1 - D_{\theta_d}(G_{\theta_g}(z))) \right]
$$
1. Gradient ascent on discriminator
$$
\max_{\theta_d} \left[ \mathbb{E}_{x \sim p_{\text{data}}} \log D_{\theta_d}(x) + \mathbb{E}_{z \sim p(z)} \log(1 - D_{\theta_d}(G_{\theta_g}(z))) \right]
$$
2. Gradient descent on generator
$$
\min_{\theta_g} \mathbb{E}_{z \sim p(z)} \log(1 - D_{\theta_d}(G_{\theta_g}(z)))
$$

最终的平衡态就是纳什均衡点

但实际上，随着训练进行，discriminator 效果提升的速度比 generator 快很多
在一开始，generator 生成的图就全都判为假的了，那梯度就会较小，难优化

![[CV导论/imgs/img14/image-7.png|317x180]]

所以把 generator 的部分换成提高生成真图的概率
这称为 **Non-saturating GAN Loss**
$$
\max_{\theta_g} \mathbb{E}_{z \sim p(z)} \log(D_{\theta_d}(G_{\theta_g}(z)))
$$

在训练 GAN 的生成器时，并没有一个明确的函数可以直接用来评估生成器性能
可以通过人工检查的方式，即定性评估，但效率低
可以用更鲁棒的定量评估，有以下指标
- Inception Score (IS) ：评价图片生成质量和多样性
- FID ：衡量生成样本分布与真实样本分布之间的距离，即相似性

FID 的流程如下
- 特征提取
    - 使用 Inception 网络（或其他 CNN 模型）的某一中间层输出，将数据样本（如图片）映射到一个特征空间中
    - 得到的特征向量捕捉了图片的语义特征，而不是像素级别的信息
- 分布假设
    - 真实样本的嵌入特征和生成样本的嵌入特征被假设为遵循多元高斯分布
    - 计算真实样本和生成样本的特征分布的均值 $\mu_r, \mu_g$ 和协方差矩阵 $\Sigma_r, \Sigma_g$
- 距离计算
    - 使用 Fréchet 距离公式计算两组高斯分布的距离
$$
FID(r, g) = \| \mu_r - \mu_g \|_2^2 + Tr(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2})
$$
- FID 值越小，表示生成样本分布与真实样本分布越接近，质量越高
- FID 对于一些不真实的变化（比如高斯模糊，扭曲）比较敏感

GAN 还展现出线性的语义关系

![[CV导论/imgs/img14/image-8.png]]

但有个问题，generator 生成图片时，可能会采取生成少数很真的图片的策略，以通过 discriminator ，这会导致 generator 生成的图片虽然比较真，但种类很少，对于训练数据的一些种类没覆盖到，这种现象称为 Mode Collapse

## Diffusion Model

如果将 VAE 看作“压缩 - 解码”的一次生成过程，那么 Diffusion Model 可以看成是“逐步去噪”的多步生成过程，从简单分布开始，逐步构建更复杂的数据分布

在 Diffusion 模型中，分为以下两个过程

前向过程：从真实数据 $x_0$ 开始，逐步加入噪声，生成一系列更模糊的中间状态直到完全噪声 $x_T$ ，这一过程是一个固定的 Markov 链，不需要学习
$$
\begin{align}
q(x_t | x_{t-1}) &= \mathcal{N}(x_t; \sqrt{1 - \beta_t} x_{t-1}, \beta_t I) \\
q(\mathbf{x}_{1:T} | \mathbf{x}_0) &= \prod_{t=1}^T q(\mathbf{x}_t | \mathbf{x}_{t-1})
\end{align}
$$
   - $\beta_t$ 是每一步中添加的噪声大小的参数
   - 最终，当 $t = T$ 时，$x_T$ 是标准高斯噪声

每一步的分布根据前一步得到，根据这个分布进行采样，即每个像素选取一个值

![[CV导论/imgs/img14/image-9.png]]

采取以下定义
$$
\bar{\alpha}_t = \prod_{s=1}^t (1 - \beta_s)
$$
那就变成了这个形式
$$
q(\mathbf{x}_t | \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_t; \sqrt{\bar{\alpha}_t} \mathbf{x}_0, (1 - \bar{\alpha}_t) \mathbf{I})
$$
这称为 Diffusion Kernel ，本质上是高斯核卷积的过程

在采样时，采取重参数化技巧
$$
\mathbf{x}_t = \sqrt{\bar{\alpha}_t} \mathbf{x}_0 + \sqrt{(1 - \bar{\alpha}_t)} \boldsymbol{\epsilon} \quad \text{where} \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})
$$

这个过程中，会逐渐趋向于高斯分布

![[CV导论/imgs/img14/image-10.png|443x224]]

逆过程：模型的目标是学习一个“去噪模型”，从随机噪声 $x_T$ 开始，逐步去掉噪声，还原到原始数据分布 $x_0$
$$
p_\theta(x_{t-1} | x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \sigma^2_t(x_t, t))
$$
- $\mu_\theta(\mathbf{x}_t, t)$ 为可学习的参数
$$
\mu_\theta(\mathbf{x}_t, t) = \frac{1}{\sqrt{1 - \beta_t}} \left( \mathbf{x}_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) \right)
$$

![[CV导论/imgs/img14/image-11.png]]

经过研究，损失函数可简化为以下形式
$$
L = \mathbb{E}_{\mathbf{x}_0, t, \boldsymbol{\epsilon}} \left[ \| \boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) \|^2 \right]
$$

总结，VAE GAN Diffusion 优缺点如下

![[CV导论/imgs/img14/image-12.png|308x232]]















