
考试形式：试题为简答题，包括但不限于
- 按要求写出计算过程和结果、解释计算结果
- 设计算法伪代码
- 写出符合条件的例句、画出符合给定条件的句法结构
- 解释某种语言现象
  
考试范围
- 基本分类问题的经典模型及评价方法
- 序列标注问题的经典模型、解码算法及评价方法
- N-gram语言模型（特点、训练、计算）
- 词汇表示基本方法（向量空间模型及计算等）
- 典型神经网络模型及其在语言模型、分类问题上的应用
- 上下文无关文法、概率型上下文无关文法、依存句法分析
- 典型句法分析算法及评价方法

# 1 分类问题

## 经典模型

**目标**：$y=\arg\max_y p(y\mid x)$ 
- **生成式模型**：学习 $p(x,y)$ ，或 $p(x\mid y)$ 和 $p(y)$ ，再用贝叶斯公式得到 $p(y\mid x)$ ，如贝叶斯模型
- **判别式模型**：直接学习 $p(y\mid x)$ ，如 Log-Linear Model 

### Naive Bayes Model

**目标**：在给定上下文 $C$ 的条件下，目标词是哪个意思的概率最高

$$
\begin{aligned}
s^* &= \arg\max_{s_k} P(s_k\mid C)
= \frac{P(C\mid s_k)P(s_k)}{P(C)}
= \arg\max_{s_k} P(C\mid s_k)P(s_k) \\
&= \arg\max_{s_k} \left(\prod_{v_x\in C} P(v_x\mid s_k)\right) P(s_k)
\end{aligned}
$$

- 候选义项集合 $S=\{s_1,s_2,...,s_n\}$ 
- 训练集 $D_w=\{(C_i,y_i)\}_{i=1}^N$ ，其中 $y_i \in S$ 
- 特征词表 $V=\{v_1,v_2,...,v_m\}$ 
- 假设各特征的出现与否相互独立

**训练方法**

$$
P(s_k)=\frac{\text{Count}(s_k)}{\text{Count}(w)}
$$

- 含义：目标词 $w$ 取 $s_k$ 义项的概率
- $\text{Count}(w)$ ：训练集中 $w$ 总共出现次数
- $\text{Count}(s_k)$ ：$w$ 取 $s_k$ 义项的次数

$$
P(v_x\mid s_k)=\frac{\text{Count}(v_x,s_k)}{\sum_{v\in V}\text{Count}(v,s_k)}
$$

- 含义：在目标词义项为 $s_k$ 的情况下，在上下文看到特征 $v_x$ 的概率
- $\text{Count}(v_x,s_k)$ ：在所有义项为 $s_k$ 的句子里，特征 $v_x$ 的出现次数
- $\sum_{v\in V}\text{Count}(v,s_k)$ ：在所有义项为 $s_k$ 的句子里，所有特征总共出现次数

**平滑**：给每个词 / 特征添加伪计数 $\alpha$ ，此时分子 $+\alpha$ ，分母 $+\alpha \lvert V\rvert$ 
- $\lvert V \rvert$ 为特征词表大小
- 引入狄利克雷分布：伪计数 $\alpha_i$ 各不相同

### Log-Linear Model

**目标**：枚举所有可能的 $y$ 候选，计算特征 $f_i(x,y)$ ，然后使用训练好的权重 $\lambda_{f_i}$ 打分，选出分数最高的 $y$ （使用 softmax 得到每个标签的概率）

$$
\begin{aligned}
y &= \arg\max_y p(y\mid x) \\
p(y\mid x) &= \frac{\exp \sum_i \lambda_{f_i(x,y)} f_i(x,y)}{\sum_{y'} \exp \sum_i \lambda_{f_i(x,y')} f_i(x,y')} = \frac{\exp(\lambda \cdot f(x,y))}{\sum_{y'} \exp(\lambda \cdot f(x,y'))}
\end{aligned}
$$

- 特征 $f_i(x,y)$ ：一系列人为定义好的函数，通常是一个指示函数，输出 0 或 1，表示某个条件是否成立
- 权重 $\lambda_{f_i}$ ：需要学习，表示特征 $f_i$ 对于分类决策的重要程度

**训练方法**：使用 MLE ，$\lambda$ 的似然函数就是在给定参数的情况下，观测到训练集 $\{(x_1,y_1),\cdots,(x_n,y_n)\}$ 的概率

$$
\begin{aligned}
L(\lambda) &= \prod_k p(y_k | x_k; \lambda) = \prod_k \frac{\exp(\lambda \cdot f(x_k, y_k))}{\sum_{y'} \exp(\lambda \cdot f(x_k, y'))} \\
LL(\lambda) &= \sum_k \log p(y_k | x_k; \lambda)
= \sum_k \lambda \cdot f(x_k, y_k) - \sum_k \log \sum_{y'} \exp(\lambda \cdot f(x_k, y')) \\
\frac{\partial LL(\lambda)}{\partial \lambda_{f_i(\cdot)}} &= \underbrace{\sum_k f_i(x_k, y_k)}_{\text{Empirical Counts}} - \underbrace{\sum_k \sum_{y'} f_i(x_k, y') p(y' | x_k; \lambda)}_{\text{Expected Counts}}
\end{aligned}
$$

- **经验计数 Empirical Counts**：特征 $f_i$ 在训练集出现的总次数
- **期望计数 Expected Counts**：特征 $f_i$ 在当前模型参数下，估计的出现次数
- 理想情况下，经验计数 = 期望计数 

**梯度下降 / 随机梯度下降**：根据当前 batch 的数据，计算经验计数和期望计数，进而得到梯度，更新权重

**正则化**
- L1 正则化：更容易把一部分参数直接压到 0 ，可以稀疏化 / 做特征选择
- L2 正则化：更倾向把所有参数都压小，但通常不变成 0 ，可以平滑收缩 / 提高泛化稳定性

## 评价标准

**TP、FP、TN、FN**
- P / N：模型的预测结果（positive / negative）
- T / F ：模型的预测是否正确（true / false）
- TP / TN：正确预测为正例 / 负例的数量
- FN：漏报，实际为正例但预测为负例的数量
- FP：误报，实际为负例但预测为正例的数量

**精确率 Precision** ：在所有被预测为正例的样本中，有多少是对的

$$
\text{Precision} = \frac{TP}{TP + FP}
$$

**召回率 Recall** ：在所有真正的正例中，模型成功找出了多少

$$
\text{Recall} = \frac{TP}{TP + FN}
$$

**F1 分数**：精确率和召回率的调和平均，二者都高时，F1 才高

$$
F_1 = \frac{2}{\frac{1}{\text{Precison}} + \frac{1}{\text{Recall}}} = \frac{2 \times \text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
$$

- **宏平均 Marco F1**：对每个类别算 F1 ，再取平均
	- 每个类别都同等重视
	- 例如在疾病诊断中，罕见病和常见病都需要被准确识别，macro-F1 能避免多数类掩盖少数类的性能
- **微平均 Micro F1**：把所有类别的 TP / FP / FN 加起来，算总体的 F1 
	- 更偏向于常见类别，因为大类别贡献更多样本

**评价协议**
- **数据集划分**
    - 训练集 Training set：用于学习模型参数
    - 验证集/开发集 Validation/Development set：用于调整超参数（如特征函数的选择）
    - 测试集：训练时完全不可见，用于最终评估模型性能
- **K-折交叉验证**：将训练集分为 $K$ 份，轮流用 $K-1$ 份训练，1 份验证，平均 $K$ 次验证结果以评估模型稳定性

# 2 N-gram 语言模型

## 训练与计算

**目标**：计算词序列 $w_1,\cdots,w_n$ 出现的概率

**计算公式**（以三元模型为例）

$$
p(w_1, w_2, ..., w_n) = p(w_1) p(w_2 \mid w_1) \prod_{i=3}^{n} p(w_i \mid w_{i-2}, w_{i-1})
$$

**训练方法**：用 MLE ，统计频率

$$
\begin{aligned}
&\text{Bigram} \quad p(w_i \mid w_{i-1}) = \frac{\text{count}(w_{i-1}, w_i)}{\text{count}(w_{i-1})} \\
&\text{Trigram} \quad p(w_i \mid w_{i-2}, w_{i-1}) = \frac{\text{count}(w_{i-2}, w_{i-1}, w_i)}{\text{count}(w_{i-2}, w_{i-1})}
\end{aligned}
$$

- **数据稀疏 data sparsity** ：当 $n$ 变得很大时，从训练集统计出的 $p(w_i \mid w_{i-1}, w_{i-2},\cdots)$ 往往只占真实语料的很小一部分，有很多搭配可能都没见过，很多条件概率都是 0 
- 一般来讲 Trigram 是一个比较实用的折中

**处理未知组合的方法**
- **回退 Back-off** ：计算 $p(w_i \mid w_{i-2}, w_{i-1},\cdots)$ 时，如果高阶 N-gram 计数不足，改为使用低阶模型
- **线性插值**：将低阶与高阶模型的概率加权平均 $p(w_i \mid w_{i-2}, w_{i-1}) = \lambda_1 p(w_i \mid w_{i-2}, w_{i-1}) + \lambda_2 p(w_i \mid w_{i-1}) + \lambda_3 p(w_i)$ 
- **加一平滑**：给每个 N-gram 计数加 1 ，问题是给没见过事件分了太多概率，把原来真正常见的事件概率压得太低

## 评价标准

**困惑度 perplexity**：模型生成文本时，预测下一个词的时候到有多懵，困惑度越小，模型越好（续写 / 预测词的能力强）

$$
\text{Perplexity}(D) = 2^{-\frac{1}{M} \sum_{s \in D} \log_2 p(s)}
$$

- 先算每个句子的概率 $p(s)$ 
- 把所有句子的概率乘起来 $\prod_{s \in D} p(s)$ 
- 用测试集总词数 $M$ 归一化 $\sqrt[M]{\prod_{s \in D} p(s)}$ 
- 取负对数 $-\frac{1}{M} \sum_{s \in D} \log_2 p(s)$ ，计算次幂

# 3 词汇表示

## 向量空间模型 VSM

**分布语义学**：词语的意义可以通过其在文本中的分布模式来刻画
- **一阶共现**：两个词经常直接相邻出现，反映搭配关系
- **二阶共现**：两个词有相似的邻居，反映同类 / 相似关系

**核心思想**：每个词表示为一个向量，向量的每个维度对应一个上下文（如共现的词、文档等）

## 构建方法

### 基于计数的方法

- 定义基础词汇表 context words 
- 选择一个窗口大小
	- 小窗口（1-3词）→ 捕捉句法相似性
	- 大窗口（5-10词）→ 捕捉语义相似性
- 统计每个目标词与上下文词的共现次数，得到稀疏的共现矩阵

```
语料：
  ... and the cute kitten purred and then ...
  ... the cute furry cat purred and miaowed ...
  ... that the small kitten miaowed and she ...
  ... the loud furry dog ran and bit ...

基础词汇：{bit, cute, furry, loud, miaowed, purred, ran, small}
kitten = [0, 1, 0, 0, 1, 1, 0, 1]
cat    = [0, 1, 1, 0, 1, 0, 0, 0]
dog    = [1, 0, 1, 1, 0, 0, 1, 0]
```

### PPMI

**问题**：原始共现计数中，高频词（如 `the`、`and`）的共现次数总是很高，但它们并不具有区分性

**点互信息 PMI** ：衡量两个词的共现是否超过随机水平

$$
\text{PMI}(x,y) = \log \frac{p(x,y)}{p(x)p(y)}
$$

- 若 $\text{PMI} > 0$ ，说明二者倾向于共现
- 若  $\text{PMI} = 0$ ，代表二者的共现是随机的
- 若  $\text{PMI} < 0$ ，说明二者不太可能共现
- 可使用拉普拉斯平滑，防止 PMI 过高

**PPMI (Positive PMI)** ：把 PMI 的负值设为 0 ，只关注相关性，不关注排斥性

![[NLP基础/img/期中/image-2.png]]

### 基于神经网络的方法

**Word2Vec**
- **跳字模型 Skip-gram** ：给定中心词，预测其上下文词
- **连续词袋模型 CBOW** ：给定上下文词，预测中心词

## 相似度计算

**余弦相似度**：将两个词向量 $u$ 和 $v$ 的夹角余弦值定义为相似度

$$
\cos (u,v) = \frac{u^T v}{\lVert u\rVert \lVert v\rVert}
$$

- 1 表示很相似，0 表示很不相似
- 问题：分布相似 ≠ 同义词，反义词往往也有非常相似的一阶分布

## TF-IDF

**词频 tf** ：某个词 $t$ 在文档 $d$ 中出现的频率

$$
\text{TF}(t,d) = \frac{\text{词 t 在文档 d 中的出现次数}}{\text{文档 d 的总词数}}
$$

- 也可使用 $\text{TF}(t,d) = \log_{10} (\text{词 t 在文档 d 中的出现次数}+1)$  

**逆文档频率 idf**：某个词 $t$ 在所有文档中的罕见程度

$$
\text{IDF}(t) = \log_{10} \frac{\text{总文档数}}{\text{包含词 t 的文档数}+1\ \text(防止除零)}
$$

**TF-IDF 权重**

$$
\text{TF-IDF}(t,d) = \text{TF}(t,d) \cdot \text{IDF}(t)
$$ 
- 如果一个词在一个文档中频繁出现（TF高），但在整个语料中很少出现（IDF高），则这个词很可能是这个文档的关键词，权重高
- 如果一个词在所有文档中都出现（如“the”），则IDF趋于0，即使TF高，最终权重也很低

## 维度约减

**问题**：基于计数的方法和 PPMI 构建的矩阵很稀疏，且维度较高
- 短向量更易用作机器学习特征
- 降维可以去噪，截断后模型能更好地泛化

**LSA (Latent Semantic Analysis)**：进行奇异值分解，保留前 $k$ 个最大的奇异值对应的特征向量

$$
A = M \cdot \text{diag}(s) \cdot C^T
$$

- $M$ 是正交阵，每行对应一个词；$C$ 是正交阵，每行对应一个文档
- 保留前 $k$ 个奇异值，$M$ 和 $C$ 也相应的调整为对应大小
	- $M_k \in \mathbb{R}^{\lvert V\rvert \times k}$ ：每行是一个词的 $k$ 维向量表示
	- $C_k \in \mathbb{R}^{\lvert D\rvert \times k}$ ：每行是一个文档的 $k$ 维向量表示

**PLSA (Probabilistic LSA)**
- 假设每个文档由一个主题混合生成 $p(\text{word}, \text{topic}|\text{document})$ 
- 对每个文档 $d$，其单词序列由 $p_{\text{topic}}(\text{topic}|d) \cdot p_{\text{word}}(\text{word}|\text{topic})$ 生成
- 类似 LDA 的前身

## 评价标准

**内在评估**
- 词相似度：计算余弦相似度
- 类比任务：通过向量运算（如 `v(China) - v(Peking) + v(Paris)`）找到相关单词

**外在评估**：将词向量作为特征输入到下游任务，看性能是否提升

# 4 神经网络

## Word Embedding

神经语言模型最关键的一步，是把词从 one-hot 变成稠密向量
- one-hot 表示的问题：高维、稀疏，而且看不出词和词之间的相似性
- 词嵌入
	- 使用嵌入矩阵 $M$ ，$m_{w_i}=w_i^T M$ 就是词 $w_i$ 的低维稠密表示
	- 好处：相似词可以有相似表示，模型就能泛化，而不是把每个词都当成完全独立的符号

## FNN

**基本思路**：取固定长度的前文，每个词进行嵌入，输入前馈网络，预测下一个词

$$
p(w_n \mid w_1,\dots,w_{n-1})
=
\text{softmax}
\left(
b+\sum_{j=1}^{n-1} m_{w_j}A_j
+
W\tanh\left(u+\sum_{j=1}^{n-1} m_{w_j}T_j\right)
\right)
$$

- $m_{w_j}$：第 $j$ 个词的 embedding 
- $T_j$：第 $j$ 个位置从输入到隐藏层的权重矩阵，表示这个位置上的词怎么影响上下文表示
- $W$：隐藏层到输出层的权重矩阵，表示上下文表示怎么变成对候选词的打分 
- $A_j$：第 $j$ 个位置从输入直接到输出层的权重矩阵，表示这个位置上的词也能直接影响输出

## CNN

**基本思路**
- 把历史词向量排成一个序列矩阵
- 用卷积核在上面滑动，提取局部特征
- 再做 pooling，压成固定长度表示
- 最后 softmax 得到概率

$$
p(w_i\mid h_{1:i-1})=\text{softmax}(FFN(\text{conv}(h_{1:i-1})))
$$

相比 FNN，它对长一点的上下文更灵活，但本质上仍更擅长抓局部信息

## RNN

**基本思路**
- 不再固定窗口，也不只是卷积扫局部片段，而是按顺序一个词一个词读进去
- 把整个历史不断压进一个隐藏状态里，再用这个状态预测下一个词

其状态递推公式是

$$
s_0=0,\quad s_i=\delta(m_{w_i}W_w+s_{i-1}W_s+b)
$$

然后用当前状态预测下一个词

$$
p(w_{i+1}\mid h_{1:i})=\text{softmax}(s_iW_p)
$$

- $m_{w_i}$：当前词的信息
- $s_{i-1}$：之前历史的压缩表示
- $s_i$：读到当前位置后的新状态

相比 FNN 和 CNN，RNN 更适合处理序列，也更自然地利用变长上下文，但普通 RNN 会有梯度消失问题，长距离依赖仍然不稳

## Transformer

## GRU

# 5 序列标注

## HMM

**目标**：序列标注
- 给定单词序列 $x_1,\dots,x_n$ ，给每个单词打标签，判断最可能的标签序列 $y_1,\cdots,y_n$ 是什么，即最大化 $p(y_1,\cdots,y_n \mid x_1,\cdots,x_n)$ 
- 根据贝叶斯公式，等价于最大化 $p(y_1,\cdots,y_n) \cdot p(x_1,\cdots,x_n \mid y_1,\cdots,y_n)$ 

**隐马尔可夫模型 HMM**
- 根据”初始概率“随机选一个标签 $y_1$ 
- 根据 $y_1$ ，用“发射概率”随机选一个词 $x_1$ 
- 根据 $y_1$ ，用“转移概率”随机选下一个标签 $y_2$ 
- 根据 $y_2$ 发射 $x_2$ ，循环下去

**重要参数**
- **初始概率** $p(y_1)$ ：句子第一个词是某个标签的概率
- **发射概率** $p(x_n\mid y_n)$ ：在给定当前标签 $y_n$ 下，看到词 $x_n$ 的概率
- **转移概率** $p(y_n\mid y_{n-1})$ ：从上一个标签 $y_{n-1}$ 转移到当前标签 $y_n$ 的概率

**训练方法**

$$
\text{初始概率} \quad p(y_1) = \frac{\mathrm{Count}(y_1\ \text{出现在句首})}{\text{句子总数}}
$$
- 统计所有句子中，第一个词的标签是 $y_1$ 的句子数，除以总句子数

$$
\text{发射概率} \quad p(x_i \mid y_i) = \frac{\mathrm{Count}(x_i, y_i)}{\mathrm{Count}(y_i)}
$$

- 统计某个单词 $x_i$​ 被标注为标签 $y_i$​ 的次数，除以标签 $y_i$​ 出现的总次数

$$
\text{转移概率} \quad p(y_i \mid y_{i-1}) = \frac{\mathrm{Count}(y_{i-1}, y_i)}{\mathrm{Count}(y_{i-1})}
$$

- 统计所有相邻标签对 $(y_{i-1}, y_i)$ 出现的次数，除以标签 $y_{i-1}$​ 出现的总次数

## 维特比解码算法

**Viterbi 算法**：一个动态规划算法，给定一个句子并用 HMM 建模，找到概率最大的标签序列

$$  
\pi(k, u, v) = \max_{w} \left[ \pi(k - 1, w, u) \cdot p(v \mid u) \cdot p(x_k \mid v) \right]  
$$

- $\pi(k, u, v)$：走到第 $k$ 个词，前一个标签为 $u$ 、当前标签为 $v$ 的所有路径中的最大概率
- $p(v \mid u)$：标签 $u \rightarrow v$ 的转移概率
- $p(x_k \mid v)$：在标签 $v$ 下生成第 $k$ 个词 $x_k$ 的发射概率

**算法过程**：从左往右，对每个位置 $k$ ，计算所有可能标签对 $(u,v)$ 的 $\pi(k, u, v)$ 值，并记录这个最大值是从哪个 $w$ 转移过来的，最后据此回溯

# 6 成分句法分析

## 上下文无关文法 CFG

**句法**：句子不是词的无序排列，而是有层次结构的（通常是树形结构）

![[NLP基础/img/期中/image.png]]

**CFG 的组成部分**
- **非终结符** $N$ ：成分类型，如名词短语 $NP$ 、动词短语 $VP$ ，是内部节点
- **终结符** $\Sigma$ ：对应到句子中的单词，是叶节点
- **开始符号** $S$ ：特殊的非终结符，表示整个句子，是根节点
- **产生式规则** $R$ ：形式为 $A \rightarrow \beta$ ，表示非终结符 $A$ 可细分为符号序列 $\beta$ （与 $A$ 的上下文无关），定义了树的分叉方式

**乔姆斯基范式 CNF** ：将产生式规则全部转换为以下两种形式
- $X \rightarrow Y\ Z$ ：一个非终结符分裂成两个非终结符
- $X \rightarrow \text{word}$ ：一个非终结符对应到一个单词

## 概率上下文无关文法 PCFG

**目标**：计算 CFG 的句法树 $t$ 出现的概率，选择最可能的

$$
p(t)=\prod_{i=1}^{n}p(A_i\to\beta_i)
$$

- 整棵树的概率等于所有用到的规则的概率的乘积

**训练方法**

$$
p(\alpha\to\beta)=\frac{\mathrm{count}(\alpha\to\beta)}{\mathrm{count}(\alpha)}
$$

- 树库：一堆标注好的句子，每个句子附带一个句法树
- $\mathrm{count}(\alpha\to\beta)$：在树库中，规则 $\alpha\to\beta$ 出现的次数
- $\mathrm{count}(\alpha)$：非终结符 $\alpha$ 出现在所有规则左侧的总次数

## CYK 算法

**CYK 算法**：一个动态规划算法，给定一个句子和一份 PCFG ，找出概率最大的树结构

$$
\pi[i,j,X]=\max_{X\to YZ\in R}\left(p(X\to YZ)\cdot\pi[i,m,Y]\cdot\pi[m,j,Z]\right)
$$

- $\pi[i,j,X]$ ：$[i,j)$ 的这段子串，能被非终结符 $X$ 覆盖的最大概率
- $m$：一个分裂位置（$i<m<j$），表示 $Y$ 覆盖 $[i,m]$ ，$Z$ 覆盖 $[m,j]$ 

**算法过程**
- 先计算对角线位置的值（对应单个单词，等于生成该词的非终结符概率）
- 记录 $\pi[i,j,X]$ 最大值对应的 $X \rightarrow Y\ Z$ 和 $m$ ，算出 $\pi[0,n,S]$ 后据此回溯

# 7 依存句法分析

## 依存句法

**句法分类**
- **成分句法**：关注句子与短语的结构关系，如 CFG 
- **依存句法**：关注词与词之间直接的语法关系 / 依赖关系，比如 `I` 是 `like` 的主语、`green` 是 `apples` 的修饰成分

**基本概念**
- **头词**：核心词，决定整个结构
- **从属词**：修饰或补充头词
- **依存关系**：从头词指向从属词的有向弧，带有依存关系标签

一些依存关系类型
- `nsubj`：名词性主语
- `obj`：直接宾语
- `advmod`：副词修饰语
- `root`：整个句子的根（通常是谓语动词）

**依存图**：节点是单词，边是依存关系
- 弱连通、有向、无环
- 单头约束：每个词只能有一个头词，但一个头词可以有多个从属词
- 投影性：如果 $w_i \rightarrow w_j$ ，则对于二者之间的单词 $w_k$ ，有 $w_i \rightarrow \cdots \rightarrow w_k$ 
	- 直观理解：把依存弧画在句子上，不会彼此交叉
	- 很多时候不假设投影性，因为需要考虑长程的依赖关系，而且一些语言的语序是很自由的

## 分析方法

### 基于转移 Transition-Based Parsing

**目标**：给每个词找到一个头词，和相应的依赖关系标签

**基本思想**
- 模拟人类的增量理解过程：从左到右逐词读入，每读一个词就尝试建立依赖关系
- 把解析过程看作一个状态机：每一步选择一个动作（转移），不断更新状态，直到处理完所有词

**核心组件**
- **状态**：一个三元组 $c = (\sigma, \beta, A)$ 
    - 栈 $\sigma$ ：已读入，但还未确认依赖关系的词
    - 缓冲区（队列） $\beta$ ：尚未读入的词
    - 集合 $A$ ：已建立的依存弧
- **动作集**
    - $\text{Shift}$ ：将缓冲区头部的词压入栈，表示读入一个词
    - $\text{Left-Arc}_k$ ：栈顶词 $w_i$ 是缓冲区第一个词 $w_j$ 的从属词，则建立 $w_j \rightarrow w_i$ 的类型为 $k$ 的依赖关系，从栈弹出 $w_i$ 
    - $\text{Right-Arc}_k$ ：栈顶词 $w_i$ 是缓冲区第一个词 $w_j$ 的头词，则建立 $w_i \rightarrow w_j$ 的类型为 $k$ 依赖关系，从栈弹出 $w_i$ ，并用 $w_i$ 替换缓冲区中的 $w_j$ 
- **预测器 Oracle** ：一个分类器，输入当前状态 $c$ ，输出动作

```
parse(sentence x = w0, w1, ..., wn):
  c = ([0], [1,2,...,n], ∅)      // 初始化
  while buffer not empty:
    action = oracle(c)            // 预测下一步动作
    c = apply_action(c, action)   // 更新状态
  return A                        // 返回所有依存弧
```

**特点**
- 贪心：每一步只取最优动作，速度快（线性时间 $O(n)$）
- 局部决策：容易受早期错误影响
- 只能处理投影结构

---

例子：`John is carefully checking the machine`

![[NLP基础/img/期中/image-1.png]]

- 初始：栈 `[ROOT]`，队列 `[John, is, carefully, checking, the, machine]`

1. **Shift** → 栈 `[ROOT, John]`，队列 `[is, ...]`
2. **Shift** → 栈 `[ROOT, John, is]`，队列 `[carefully, ...]`
3. **Shift** → 栈 `[ROOT, John, is, carefully]`，队列 `[checking, ...]`
4. **Left-Arc_advmod** → `carefully` 是 `checking` 的副词修饰，建立弧，弹出 `carefully` → 栈 `[ROOT, John, is]`，队列 `[checking, ...]`
5. **Left-Arc_aux** → 建立 `is → checking` 的弧，弹出 `is` → 栈 `[ROOT, John]`，队列 `[checking, ...]`
6. **Left-Arc_nsubj** → 建立 `John → checking` 的弧，弹出 `John` → 栈 `[ROOT]`，队列 `[checking, ...]`
7. **Shift** → 栈 `[ROOT, checking]`，队列 `[the, machine]`
8. **Shift** → 栈 `[ROOT, checking, the]`，队列 `[machine]`
9. **Left-Arc_det** → 建立 `the → machine` 的弧 → 栈 `[ROOT, checking]`，队列 `[machine]`
10. **Right-Arc_obj** → 建立 `checking → machine` 的弧，且用 `checking` 替换队列头 → 栈 `[ROOT]`，队列 `[checking]`
11. **Right-Arc_root** → 建立 `ROOT → checking` 的弧，用 `checking` 替换 → 栈 `[]`，队列 `[checking]`
12. **Shift** → 栈 `[ROOT]`，队列 `[]` → 完成

### 基于图 Graph-Based Parsing

**思路**：用模型给每个依赖弧打分，作为边权，而后找到依存图的最大生成树

![[NLP基础/img/期中/image-3.png]]

**特点**
- 全局最优：考虑了所有弧的分数
- 比转移方法慢，时间复杂度 $O(n^3)$ 
- 可以处理非投影结构

## 评价标准

- **UAS（未标记依存准确率）**：正确预测头词的词的比例
- **LAS（带标记依存准确率）**：正确预测头词和关系标签的词的比例
- **句子准确率**：整个句子完全正确的比例


