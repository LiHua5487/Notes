---
date: 2025-06-04
tags:
  - CV
---

# Generative Model

分类模型通常回答“这是什么”，**生成模型 generative model**则希望生成与训练数据相似的新样本，比如一张图片，可以把这个过程理解为

- 训练：根据真实数据，学习一个概率模型 $p_\theta(x)$，使其接近真实数据分布 $p_{data}(x)$
- 生成：从学到的模型中采样，得到一个新的 $x$

$x$ 可以是整张图片，把各个像素值放在一起，就得到高维向量，分布描述的是哪些像素组合更容易出现，而不只是每个像素单独有多亮

连续变量用概率密度描述，密度可以大于 1，但一个范围内的概率需要通过积分得到，仍在 $0\sim1$ 之间；离散像素值也可以用离散概率建模，下面根据具体模型区分

![[CV导论/imgs/img14/image.png|466x275]]

在带类别标签的情况中，判别式模型常直接学习 $p(y\mid x)$，生成式方法可以学习 $p(x)$ 或 $p(x,y)$，即描述数据本身或数据与标签如何共同出现

按概率密度的处理方式，可以分为

- **显式密度 explicit density**：明确给出概率模型
  - tractable：密度或似然能够直接计算，比如一些自回归模型
  - approximate：采用近似推断或变分下界等方式训练，比如 VAE、DDPM
- **隐式模型 implicit model**：能够采样，但通常不直接计算样本密度，比如 GAN

一个简单的显式建模方式是把联合概率按链式法则拆开

$$
\begin{aligned}
p(x)&=p(x_1,x_2,\ldots,x_n) \\
&=\prod_{i=1}^n p(x_i\mid x_1,\ldots,x_{i-1})
\end{aligned}
$$

指定像素或 token 的顺序，就可以根据之前已经生成的部分预测下一个，这称为**自回归 autoregressive**，但逐个生成会限制采样速度

## Variational Autoencoders (VAE)

### From Autoencoder to a Probability Model

普通自编码器 autoencoder 先用 encoder 将输入 $x$ 压成潜在表示 latent representation $z$，再用 decoder 尽量重构原来的输入

![[CV导论/imgs/img14/image-1.png|327x211]]

训练完成后，给一个输入就能得到相应的重构，但如果要凭空生成，还需要知道应该从哪里取 $z$，随便取一个向量不一定落在训练过的有效区域

**变分自编码器 VAE**给这个过程建立概率模型：先从简单的先验分布 $p(z)$ 采样潜变量，再由 $p_\theta(x\mid z)$ 生成数据，常设 $p(z)=\mathcal N(0,I)$，即各维独立的标准高斯分布

![[CV导论/imgs/img14/image-2.png|348x198]]

对于一个图像 $x$ ，模型生成它的概率为

$$
p_\theta(x)=\int p(z)p_\theta(x\mid z)\,dz
$$

这个积分的含义是：所有可能的 $z$ 都有机会产生 $x$，把这些可能性按先验概率加起来，得到 $x$ 的**边缘似然 marginal likelihood**，训练希望它在真实数据上尽量大

但高维积分通常难以精确计算，直接从先验反复采样估计也可能效率很低，根据贝叶斯公式还可以写成

$$
p_\theta(x)=\frac{p_\theta(x,z)}{p_\theta(z\mid x)}=\frac{p(z)p_\theta(x\mid z)}{p_\theta(z\mid x)}
$$

这里的**后验 posterior** $p_\theta(z\mid x)$ 表示“已经看到这张图后，哪些潜变量更可能产生它”，但它同样难以计算，所以用 encoder 输出一个容易采样的分布 $q_\phi(z\mid x)$ 去近似，常选对角高斯，由网络预测均值 $\mu_{z\mid x}$ 和标准差 $\sigma_{z\mid x}$

![[CV导论/imgs/img14/image-3.png]]

### Evidence Lower Bound (ELBO)

我们想最大化 $\log p_\theta(x)$，但它难以直接计算，先进行一下变形

![[CV导论/imgs/img14/image-4.png]]

$$
\begin{aligned}
\log p_\theta(x)
&=\mathcal L_{ELBO}(x)+D_{KL}\big(q_\phi(z\mid x)\|p_\theta(z\mid x)\big)\geq\mathcal L_{ELBO}(x) \\
\mathcal L_{ELBO}(x)
&=\mathbb E_{z\sim q_\phi(z\mid x)}\left[\log p_\theta(x\mid z)\right]
-D_{KL}\big(q_\phi(z\mid x)\|p(z)\big)
\end{aligned}
$$

$\mathcal L_{ELBO}(x)$ 称为**证据下界 evidence lower bound (ELBO)**，将其作为训练的最大化目标
- 重构项：编码出来的潜变量 $z$ ，尽可能让 decoder 把原图 $x$ 重建出来
- KL 项：让 encoder 的输出分布不要过分偏离先验，使生成时从先验采样也有机会落到合适区域

其中重构项的期望可以用从 $q_\phi(z\mid x)$ 中采出的少量样本近似，取一个样本时得到

$$
\widehat{\mathcal L}_{ELBO}(x)=\log p_\theta(x\mid z)-D_{KL}\big(q_\phi(z\mid x)\|p(z)\big),\quad z\sim q_\phi(z\mid x)
$$

对于高斯 encoder 与标准高斯先验，KL 项还可以直接计算

$$
D_{KL}\big(\mathcal N(\mu,\operatorname{diag}(\sigma^2))\|\mathcal N(0,I)\big)
=\frac12\sum_j\left(\mu_j^2+\sigma_j^2-1-\log\sigma_j^2\right)
$$

### Reparameterization

从 encoder 到 decoder 中间需要经过一个采样的环节，这没法直接反向传播，可以用**重参数化技巧 reparameterization trick**

$$
z=\mu_{z\mid x}+\sigma_{z\mid x}\odot\epsilon,\quad\epsilon\sim\mathcal N(0,I)
$$

这样梯度就能沿着下面的链路进行传播

$$
\mathcal L\rightarrow\hat x\rightarrow z \rightarrow(\mu,\sigma)\rightarrow\phi
$$

## Generative Adversarial Networks (GAN)

### Generator and Discriminator

真实图片位于复杂的高维数据分布中，很难直接写出密度公式，但我们可以从简单分布中采样噪声 $z$，再训练一个网络 $G_\theta(z)$ 将其变成图片

![[CV导论/imgs/img14/image-6.png|334x202]]

与 VAE 一样，这里也能通过非线性网络把简单潜变量分布变成复杂图像分布，GAN 的关键在于如何训练生成器，而不是要求显式算出每张图的概率

如果只拿某一张固定图片与生成结果做像素误差，就不能充分衡量整个生成分布是否真实；平方误差在多种合理输出之间还可能偏向平均结果，产生模糊，所以引入另一个模型来评价

**生成器 generator** $G$负责生成图片，**判别器 discriminator** $D$输出“这张图来自真实数据”的概率，训练时真实样本标为真、生成样本标为假，$D$ 学习区分两者，$G$ 则学习让自己的输出更容易被认为是真的

![[CV导论/imgs/img14/image-5.png]]

### Adversarial Training

原始 GAN 采用下面的极小极大 minimax 目标

$$
\min_{\theta_g}\max_{\theta_d}
\left[\mathbb E_{x\sim p_{data}}\log D_{\theta_d}(x)
+\mathbb E_{z\sim p(z)}\log\big(1-D_{\theta_d}(G_{\theta_g}(z))\big)\right]
$$

两个网络交替更新

1. 更新 $D$：对真实数据输出接近 1，对生成数据输出接近 0，因此最大化上面的判别目标
2. 更新 $G$：暂时固定 $D$，让生成样本的判别结果变大，原始形式最小化 $\mathbb E_z\log(1-D(G(z)))$

理想情况下，当生成分布与真实分布相同，最优判别器无法区分两者，输出为 $1/2$，这对应理论上的平衡；实际训练不保证达到这个状态

训练早期，判别器可能很容易看出生成图是假的，即 $D(G(z))$ 接近 0，原始生成器目标容易饱和，给生成器的有效梯度太弱

![[CV导论/imgs/img14/image-7.png|317x180]]

所以常改用**非饱和生成器损失 non-saturating GAN loss**，直接提高生成样本被判为真的对数概率

$$
\max_{\theta_g}\mathbb E_{z\sim p(z)}\log D_{\theta_d}(G_{\theta_g}(z))
$$

也就是最小化它的负值，目标仍是让生成结果更真实，但早期梯度通常更有用

### Evaluation and Mode Collapse

生成器没有一个简单的分类准确率可以直接代表质量，可以人工查看，也可以使用定量指标

- **Inception Score (IS)**：综合考察生成图像的预测类别是否明确、整体类别是否多样
- **FID**：比较真实样本与生成样本在特征空间中的分布差异

计算 FID 时，先通过预定的 Inception 网络等特征提取器将图片变成向量，再分别统计真实、生成特征的均值和协方差，用高斯近似比较两者

$$
\operatorname{FID}(r,g)=\|\mu_r-\mu_g\|_2^2+
\operatorname{tr}\left(\Sigma_r+\Sigma_g-2(\Sigma_r^{1/2}\Sigma_g\Sigma_r^{1/2})^{1/2}\right)
$$

协方差记录特征各维如何共同变化，$\operatorname{tr}$ 是矩阵对角线之和，FID 越低通常表示特征分布越接近，比较时应使用一致的样本数、预处理和特征网络，单个分数也不能覆盖生成质量的所有方面

GAN 的潜空间有时会表现出近似线性的语义关系，比如对特征向量做加减，生成图像也出现对应变化

![[CV导论/imgs/img14/image-8.png]]

但生成器可能只会产生少数几类很像真的图片，漏掉真实分布中的其他类型，这称为**模式崩塌 mode collapse**，所以评价时既要看单张质量，也要看多样性和覆盖程度

## Diffusion Model

**扩散模型 diffusion model**把生成过程拆成许多较小的步骤：先定义如何逐渐给真实图片加噪，再训练网络学习反方向的去噪过程，下面以 **DDPM (Denoising Diffusion Probabilistic Models)**为例

也可以从层级潜变量模型的角度理解 DDPM：每个噪声级别都是一个潜在状态，前向加噪过程固定，反向过程需要学习，和 VAE 一样可以通过变分目标建立训练依据

### Forward Process

从真实图像 $x_0$ 开始，每一步稍微缩小已有信号，再加入高斯噪声，得到 $x_1,\ldots,x_T$，这是一条固定的 Markov 链，即下一步只依赖当前状态

$$
\begin{aligned}
q(x_t\mid x_{t-1})&=\mathcal N\big(x_t;\sqrt{1-\beta_t}\,x_{t-1},\beta_t I\big) \\
q(x_{1:T}\mid x_0)&=\prod_{t=1}^Tq(x_t\mid x_{t-1})
\end{aligned}
$$

$\beta_t$ 是第 $t$ 步加入的噪声方差，预先给定，$I$ 表示各维使用同样的方差且噪声相互独立，噪声步数和调度合适时，$x_T$ 会接近标准高斯噪声

![[CV导论/imgs/img14/image-9.png]]

定义 $\alpha_t=1-\beta_t$、$\bar\alpha_t=\prod_{s=1}^t\alpha_s$，多步高斯变换可以合并，直接得到第 $t$ 步的条件分布

$$
q(x_t\mid x_0)=\mathcal N\big(x_t;\sqrt{\bar\alpha_t}\,x_0,(1-\bar\alpha_t)I\big)
$$

这个高斯条件分布也称为 diffusion kernel，这里的加噪作用于图像向量的数据空间，不是在图像上滑动高斯滤波窗口做空间模糊

采样时同样可以重参数化

$$
x_t=\sqrt{\bar\alpha_t}\,x_0+\sqrt{1-\bar\alpha_t}\,\epsilon,
\quad\epsilon\sim\mathcal N(0,I)
$$

因此训练时想得到任意噪声级别的 $x_t$，不需要真的从第 1 步一路加噪到第 $t$ 步，抽一个 $t$、一份噪声就能直接构造

![[CV导论/imgs/img14/image-10.png|443x224]]

### Reverse Process

反向过程从高斯噪声出发，逐步生成更干净的状态，最后得到图像，DDPM 用高斯分布参数化每一步的反向转移

$$
p_\theta(x_{t-1}\mid x_t)=\mathcal N\big(x_{t-1};\mu_\theta(x_t,t),\sigma_t^2 I\big)
$$

方差可以采用预定方案，也可以学习，均值则可以通过预测前向累计噪声来得到，训练网络 $\epsilon_\theta(x_t,t)$，让它根据带噪图像和时间步，预测构造 $x_t$ 时用的那份 $\epsilon$

$$
\mu_\theta(x_t,t)=\frac1{\sqrt{\alpha_t}}
\left(x_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\epsilon_\theta(x_t,t)\right)
$$

已知 $x_0$ 时，前向过程对应的条件后验 $q(x_{t-1}\mid x_t,x_0)$ 可以精确写成高斯分布，上式利用其均值的形式，将真实噪声换成网络预测；实际生成时不知道 $x_0$，所以需要学会预测，不能认为任意加噪分布的精确逆过程必然都是高斯

![[CV导论/imgs/img14/image-11.png]]

DDPM 常用的简化训练目标是噪声预测的平方误差

$$
L_{simple}=\mathbb E_{x_0,t,\epsilon}
\left[\|\epsilon-\epsilon_\theta(x_t,t)\|_2^2\right]
$$

训练一次的过程是：抽取真实图片 $x_0$、时间步 $t$ 和噪声 $\epsilon$，构造 $x_t$，让模型预测这份噪声，再反向传播更新参数；生成时则先采样 $x_T$，根据模型给出的反向分布一步步采样，直到得到 $x_0$

VAE 通过潜变量分布和重构目标学习，GAN 通过生成器与判别器对抗学习，diffusion 则通过多步去噪学习，各自有不同的训练稳定性、采样速度和质量取舍

![[CV导论/imgs/img14/image-12.png|308x232]]
