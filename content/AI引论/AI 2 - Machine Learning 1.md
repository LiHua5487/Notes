---
date: 2025-03-24
tags:
  - AI
---
# 基本概念

- 模型(Model)：一个包含参数(parameters) 的函数 
- 标签(Label)：要预测的类别或数值
- 模型训练/学习(training/learning)：通过调整模型参数来拟合 (fit) 训练数据 (training data)

- 模型分类
![[AIimg/img2/image.png]]

- 非参数化模型：
	- 模型不能被有限参数定义，或不包含参数
	- 需要保留训练样本，以对测试样本做出预测 
- 参数化模型
	- 模型包含可训练的参数，通过拟合训练数据来估算模型参数 
	- 训练好模型参数后，可以丢弃训练数据，仅依靠模型参数去预测新样本 

- 回归(regression): 标签 (label) 是连续 (continuous) 值 （如 预测房价） 
- 分类(classification): 标签 (label) 是离散 (discrete) 值 （如 判断狗的品种）

- 训练误差(training error)：在训练集上的平均误差 
	- 通过最小化训练误差来训练模型 
	- 对分类问题，通常用错误率来衡量训练误差
- 测试误差(test error)：在测试集上的平均误差
	- 训练完成后，用来真正衡量模型在新数据上的好坏 
	- 衡量模型的泛化(generalization) 能力 
	- 训练误差低不代表测试误差一定也低（过拟合） 
- 训练集(training set)、测试集(test set) 划分 
	- 给定全部的数据，按一定比例随机划分为训练集和测试集
	- 按照时间顺序排列，再按先后划分为训练集和测试集

- 过拟合(overfitting)：测试误差远远大于训练误差 
	- 错把训练样本中找到的特殊规律当做了普遍规律，应避免这种现象
- 欠拟合(underfitting)：训练完成后，训练误差仍然很大
	- 说明连训练样本都没有拟合好

训练过程中一般涉及一些超参数(hyperparameters)
- 梯度下降的学习率 𝛼，正则化项的系数 𝜆 等
- 此外，使用哪种模型、哪种损失函数、哪种正则化等也可看作超参数 

使用验证集(validation set) 来选择最优超参数
- 在训练集、测试集之外引入验证集，通常按照训练集/验证集/测试集分别 80%/10%/10%的比例划分 
- 对每一组超参数的组合，在训练集上训练参数，在验证集上验证模型误差
- 遍历所有可能的超参数组合，选择在验证集上误差最小的模型和超参数
- 使用选定的模型和超参数在测试集上最终测试模型误差，估计模型泛化能力

# K-折交叉验证

有时总数据量比较少，10%的验证集不足以准确地验证模型性能

K-折交叉验证
- 将测试集外的所有数据随机分为 K 份 
- 对一组给定的超参数组合： 
	- 每次使用 K-1 份数据训练模型，用1份验证模型误差 
	- 将K次验证的模型误差取平均来衡量该组超参数的好坏 
	- 遍历所有可能的超参数组合 
- 返回平均误差最低的超参数组合，在测试集上最终测试模型性能

# Empirical Risk Minimization (ERM)

- 确定模型 $f(x)$
- 确定损失函数 $L(f(x),y)$
- 在训练集上最小化损失函数均值

$$
\min_{\theta}\frac{1}{n} \sum_{i \in [n]} L(f(x_i), y_i)
$$

一般可以用GD优化，对于监督式学习，这个流程是通用的，具体差别在于model与Loss的选取

# k-临近算法

对于一个测试样本，用训练样本中距离它最近的k个样本中占多数的标签 来预测测试样本

![[AIimg/img2/image-1.png|700x233]]

加权的 k-NN ：对于不同的距离，可以对应不同的权重，距离越近贡献越大，计算不同类别的权重和，取大的一个

- 优点
	- 没有任何内置参数，不需要训练 
	- 只需要一个距离函数即可（默认为欧氏距离）

- 缺点
	- 需要存储所有训练样本 
	- 在测试时需要计算测试样本到所有训练样本的距离 
	- 有时很难找到一个好的距离函数，或不一定能准确反映相似度
	- 在高维度时，会发生**维度灾难 Curse of Dimensionality**

![[AI引论/AIimg/img2/image-4.png]]

# 线性回归

模型

$$
f(x) = w^T x + b
$$

- 输入: $x \in \mathbb{R}^d$
- 参数: $w \in \mathbb{R}^d, b \in \mathbb{R}$，权重 (weight) 和 偏置 (bias)
- 输出: $f(x) \in \mathbb{R}$

损失函数
- 平方损失函数 (squared loss, L2 loss)

$$
J(w, b) = \frac{1}{n} \sum_{i \in [n]} L(f(x_i), y_i) = \frac{1}{n} \sum_{i \in [n]} (w^T x_i + b - y_i)^2
$$

- 优化方法：梯度下降训练
- 性质：凸函数（局部最小值即为全局最小值）

对于 L2 Loss，有

$$
\begin{aligned}
J(w, b) = \frac{1}{n} \sum_{i \in [n]} L(f(x_i), y_i) &= \frac{1}{n} \sum_{i \in [n]} (w^T x_i + b - y_i)^2 = \frac{1}{n} \sum_{i \in [n]} e_i^2 \\
where\ e_i &= w^T x_i + b - y_i
\end{aligned}
$$

注意到，L2 Loss 是凸函数，其梯度

$$
\begin{aligned}
\frac{\partial J(w, b)}{\partial w} &= \frac{2}{n} \sum_{i \in [n]} (w^T x_i + b - y_i) x_i \in \mathbb{R}^d \\
\frac{\partial J(w, b)}{\partial b} &= \frac{2}{n} \sum_{i \in [n]} (w^T x_i + b - y_i) \in \mathbb{R}
\end{aligned}
$$

据此更新参数

$$
\begin{aligned}
w &\leftarrow w - \alpha \cdot \frac{2}{n} \sum_{i \in [n]} e_i x_i \\
b &\leftarrow b - \alpha \cdot \frac{2}{n} \sum_{i \in [n]} e_i
\end{aligned}
$$

训练过程：
1. 给定训练数据 $\{(x_1, y_1), (x_2, y_2), ..., (x_n, y_n)\}$，学习率 $\alpha$
2. 初始化模型参数 $w, b$
3. 训练模型：
	- 计算模型预测值 
	  $\hat{y}_i = w^T x_i + b, \ \forall i \in [n]$  
	- 计算平方损失函数  
	  $J(w, b) = \frac{1}{n} \sum_{i \in [n]} (\hat{y}_i - y_i)^2$  
	- 计算梯度  
	  $\frac{\partial J(w, b)}{\partial w} = \frac{2}{n} \sum_{i \in [n]} (\hat{y}_i - y_i) x_i$  $\frac{\partial J(w, b)}{\partial b} = \frac{2}{n} \sum_{i \in [n]} (\hat{y}_i - y_i)$  
	- 梯度下降更新 $w, b$  
	  $w \gets w - \alpha \cdot \frac{\partial J(w, b)}{\partial w}, \ b \gets b - \alpha \cdot \frac{\partial J(w, b)}{\partial b}$  
	- 重复以上步骤，直到损失函数不再下降或达到预设迭代次数  
4. 给定新数据 $x$，使用训练好的模型预测其标签  $\hat{y} = w^T x + b$

# 逻辑回归

- 应用：处理二分类问题
- 模型：线性模型 $f(x) = w^Tx+b$  ，最后用激活函数化为概率
- 损失函数：交叉熵损失 cross entropy loss / logistic loss

## 二分类问题

标签只有两种
  - $y \in \{-1, 1\}$，$-1$ 代表负类 (negative class)，$1$ 代表正类 (positive class)
  - 例如垃圾邮件识别中，$1$ 代表垃圾邮件，$-1$ 代表正常邮件

一般不直接让 $f(x) \in \mathbb{R}$ 拟合 $y \in \{-1,1\}$
  - 为了将实数输出 $f(x)$ 转换为类别 $\{-1, 1\}$，通常采用 $\text{sign}$ 函数：

$$
\text{sign}(f(x)) = 
\begin{cases} 
1, & \text{if } f(x) > 0 \\
-1, & \text{if } f(x) < 0
\end{cases}
$$

使用什么损失函数？
  - 最直接的目标是最小化分类错误数，采用**零一损失函数 (zero-one loss)**：

$$
 L(f(x), y) = 
\begin{cases} 
 0, & \text{if } \text{sign}(f(x)) = y \quad \text{(分类正确)} \\
1, & \text{if } \text{sign}(f(x)) = -y \quad \text{(分类错误)} \end{cases}
$$

  或等价地写作：

$$
L(f(x), y) = 
\begin{cases} 
0, & \text{if } y \cdot f(x) \geq 0 \\
1, & \text{if } y \cdot f(x) < 0
\end{cases}
$$

![[AIimg/img2/image-2.png|449x175]]

- 问题：零一损失函数的缺点
  - 零一损失函数是阶跃函数，不可微且不连续 
  - 梯度下降无法优化：
    - 在 $f(x) = 0$ 时不可微
    - 其余处梯度都为 $0$，无法提供下降方向

## MLE 与 交叉熵损失

对观测数据进行 (条件) 概率建模
- 对机器学习，每个观测数据即一个训练样本
- 对判别式模型，我们只建模 $p(y|x; \theta)$, $\theta$ 为模型参数
- 通过最大化观测数据在给定概率模型下的似然（例如：把训练样本预测为正确标签的**概率**）来估计模型参数
如果训练样本相互独立（独立同分布假设），则最大似然估计可写为  

$$
\max_{\theta} \prod_{i \in [n]} p(y = y_i | x = x_i; \theta)
$$

或简写为  

$$
\max_{\theta} \prod_{i \in [n]} p(y_i | x_i; \theta)
$$

但是，大量概率连乘容易造成数值超出计算精度
所以采用最大化**对数似然 (log-likelihood)**  

$$
\max_{\theta} \log\left(\prod_{i \in [n]} p(y_i | x_i; \theta)\right) \iff \max_{\theta} \sum_{i \in [n]} \log(p(y_i | x_i; \theta))
$$

---

以抛硬币为例，假设抛 n 次，正面有 m 次，估计正面概率 p

$$
\begin{aligned}
Likelihood &= p^m(1-p)^{n-m} \\
log-L &= mlogp + (n-m)log(1-p)
\end{aligned}
$$

最大化 $log-L$，求其梯度为0的点，得到 $p=\frac{m}{n}$ ，经检验是全局最大值

---

对于逻辑回归，先把线性模型 $f(x)=w^T+b$ 转化为概率
采用 sigmoid 函数归一化

$$
\begin{aligned}
\sigma(z) &= \frac{1}{1 + e^{-z}} \\
1-\sigma(z) &= \frac{e^{-z}}{1 + e^{-z}} = \frac{1}{1 + e^{z}} = \sigma(-z)
\end{aligned}
$$

无论 y 取 1 还是 -1，均有

$$
\begin{aligned}
p(y = 1 | x; \theta) &= \sigma(f(x)) = \sigma(y \cdot f(x)) \\
p(y = -1 | x; \theta) &= 1 - \sigma(f(x)) = \sigma(-f(x)) = \sigma(y \cdot f(x))
\end{aligned}
$$

其中 $\theta$ 为 $(w,b)$ ，故可采用下式作为概率

$$p(y | x; \theta)=\sigma(y \cdot f(x))$$

则 log-likelihood 可表示为 

$$
\begin{aligned*}
\max_{w, b} \sum_{i \in [n]} \log \big[p(y_i | x_i; w, b)\big] 
&= \sum_{i \in [n]} \log \big[\sigma(y_i \cdot f(x_i))\big] \\
& = \sum_{i \in [n]} \log \big[\sigma(y_i(w^T x_i + b))\big] \\
&= \sum_{i \in [n]} \log \Bigg[\frac{1}{1 + e^{-y_i (w^T x_i + b)}}\Bigg] \\[10pt]
&= -\sum_{i \in [n]} \log \big[1 + e^{-y_i (w^T x_i + b)}\big] 
\end{aligned*}
$$

转化为 ERM 的形式，写成损失函数，称为 **交叉熵损失**

$$
L(f(x_i),y_i) = \frac{1}{n} \sum_{i \in [n]} \log \big[1 + e^{-y_i (w^T x_i + b)}\big]
$$

- 这个式子是交叉熵损失对于二分类问题的情况
注意到，交叉熵损失函数连续可微，且为凸函数，其梯度

$$
\begin{aligned*}
    \frac{\partial J(w, b)}{\partial w} 
    &= -\frac{1}{n} \sum_{i \in [n]} \frac{e^{-y_i (w^T x_i + b)}}{1 + e^{-y_i (w^T x_i + b)}} \cdot y_i x_i \\[10pt]
    &= -\frac{1}{n} \sum_{i \in [n]} \Big[1 - p(y_i | x_i; w, b)\Big] y_i x_i \\[15pt]
    \frac{\partial J(w, b)}{\partial b} 
    &= -\frac{1}{n} \sum_{i \in [n]} \frac{e^{-y_i (w^T x_i + b)}}{1 + e^{-y_i (w^T x_i + b)}} \cdot y_i \\[10pt]
    &= -\frac{1}{n} \sum_{i \in [n]} \Big[1 - p(y_i | x_i; w, b)\Big] y_i
\end{aligned*}
$$

直观地来看，样本 $i$ 预测为其真实标签的概率越接近1（说明已经充分拟合），它对梯度的贡献越小，即不太需要调整

注：逻辑回归仍是线性模型，只是训练时使用 sigmoid 函数进行非线性处理

# Softmax 回归

对 K 分类问题，训练 K 个二分类器（如 逻辑回归）
第 k 个二分类器将第 k 类当成正类， 其余所有类别都当成负类

问题是，当类别数K非常大时，分别训练 K 个二分类器代价太高
而且，K个二分类器互相独立，无法统一成一个模型

采用 Softmax 回归
- 应用：K分类问题： $y \in \{1, 2, \dots, K\} = [K], x \in \mathbb{R}^d$
- 模型：共同训练 K 个模型 $f_k(x) \in \mathbb{R}, k \in [K]$

训练时，要将模型输出 $f_k(x)$ 转化为取第 $k$ 类的概率
- 不能使用 sigmoid 函数，因为需满足归一化条件

$$\sum_{k \in [K]} p(y = k | x) = 1$$

- 解决方法：使用 softmax 函数

$$
p(y = k | x) = \frac{e^{f_k(x)}}{\sum_{j \in [K]} e^{f_j(x)}}
$$

- e 指数的放大效应会使得如果 $f_k(x) \gg f_j(x), \forall j \neq k$，则 $p(y = k | x) \approx 1$

假设 $f_k(x)$ 的参数为 $\theta_k$，最大化训练集 $\{(x_i, y_i) | i \in [n]\}$ 的对数似然： $$ \max_{\{\theta_k\}} \sum_{i \in [n]} \log p(y_i | x_i) = \sum_{i \in [n]} \log \left[ \frac{e^{f_{y_i}(x_i)}}{\sum_{j \in [K]} e^{f_j(x_i)}} \right] $$等价于最小化交叉熵损失： 

$$ \min_{\{\theta_k\}} -\frac{1}{n} \sum_{i \in [n]} \log \left[ \frac{e^{f_{y_i}(x_i)}}{\sum_{j \in [K]} e^{f_j(x_i)}} \right] = \frac{1}{n} \sum_{i \in [n]} \left( \log \left[ \sum_{j \in [K]} e^{f_j(x_i)} \right] - f_{y_i}(x_i) \right) $$

# 正则化

实际中在损失函数后加一项正则化项一起优化，以防止过拟合

$$
\min_{f} \frac{1}{n} \sum_{i \in [n]} L(f(x_i), y_i) + \lambda \cdot R(f)
$$

- $\lambda \cdot R(f)$ 称为**正则化项** (regularization term)，用于惩罚过于复杂的模型
- $\lambda > 0$ 是一个超参数 (hyperparameter)，需要预先指定，不随参数优化

常见过拟合原因：

1. 某几个特征维度 $j$ 支配 (dominate) 了预测，即这些维度的权重 $w_j$ 过大

- 采用 **L2 正则化** 

$$R(f) = \|w\|_2^2 = w^T w = \sum_{j \in [d]} w_j^2$$

- 作用：$w_j^2$ 会放大较大的权重，用于惩罚少数过大的权重维度，使分配更平均

- 训练方法：**岭回归 (Ridge Regression)**

$$ \min_{w, b} \frac{1}{n} \sum_{i \in [n]} (w^T x_i + b - y_i)^2 + \lambda \|w\|^2 $$


$$ \frac{\partial J(w, b)}{\partial w} = \frac{2}{n} \sum_{i \in [n]} (w^T x_i + b - y_i)x_i + 2\lambda w \in \mathbb{R}^d $$

2. 输入数据中存在大量没用的特征维度，但仍然赋予了它们非零的权重

- 采用 **L1 正则化** 

$$R(f) = \|w\|_1 = \sum_{j \in [d]} |w_j|$$

- 作用：鼓励稀疏的 $w$，即 $w$ 中大部分维度为零，仅有少数维度非零

- 训练方法：**Lasso回归**

$$ \min_{w, b} \frac{1}{n} \sum_{i \in [n]} (w^T x_i + b - y_i)^2 + \lambda \|w\|_1 $$

![[AIimg/img2/image-3.png|442x323]]


逻辑回归 (完整形式)
- 交叉熵损失 + L2 正则化

$$
\min_{w, b} \frac{1}{n} \sum_{i \in [n]} \log(1 + e^{-y_i(w^T x_i + b)}) + \lambda \|w\|^2
$$

- 梯度

$$
\frac{\partial J(w, b)}{\partial w} = -\frac{1}{n} \sum_{i \in [n]} [1 - p(y_i | x_i; w, b)] y_i x_i + 2\lambda w \quad \in \mathbb{R}^d
$$




