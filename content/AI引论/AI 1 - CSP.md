---
date: 2025-03-06
tags:
  - AI
---
# CSP 约束满足问题

CSP是一类特殊的搜索问题，假设每个状态可以由一系列变量的取值的组合，即 $(x_1,...,x_n)$ 表示，其中每个变量有一定的取值范围 $x_i \in D$ ，目标是在给定的约束下，求 $x_i$ 可能的取值组合
- 状态：一些变量的赋值 
- 目标：所有变量都有赋值， 并且满足约束
- 动作：对一个还没有赋值的 变量进行赋值

Example 着色问题

![[AIimg/img1/image.png|424x333]]

- 变量：不同区域 $WA, NT, ...$ 的颜色
- 域：$D\ =\ \{R,\ G,\ B\}$
- 约束：相邻区域颜色不同
- 目标：符合约束条件的染色方式

对于CSP，回溯搜索是一种通用的解法

## Backtracking Search 回溯搜索

在DFS的基础上，进行以下改进

- 改进1：一次（每层）只考虑一个未赋值的变量，变量赋值的顺序不重要，即先 WA = red 再 NT = green 等价于 先 NT = green 再 WA = red
- 改进2：每一步都判断约束的满足情况，即只考虑与前序赋值不矛盾的可能

**回溯搜索 = DFS + 改进1 + 改进2**

对于回溯算法，怎么才能提高其速度？
- 顺序
	- 什么变量应该下一个赋值
	- 每个变量应该先试什么值
- 筛选
	我们可以提早识别最终的失败吗

### 顺序

#### MRV 最少剩余值启发

赋什么变量：选择有最多约束，即剩余可能值最少的变量赋值

![[AIimg/img1/image-1.png]]

#### LCV 最少约束值

赋什么值：当一个变量要被赋值的时候，选给剩下的变量留下最多可能的值

![[AIimg/img1/image-2.png]]

### 筛选

#### Forward Checking 前向检查 

对于后续的一些取值，加到现有赋值中就会违反一些约束，这些取值就可以去掉

#### Constraint Propagation 约束传递

提前检查将信息从已赋值的传递到未赋值的，但还是不能提前发现所有失败

![[AIimg/img1/image-3.png]]

如图，NT与SA不能同取蓝色，但FC没有提前发现
为此，使用约束传递，即从现有约束推理到其他的可能存在的约束

引入 **Consistency** 一致性 的概念：
考虑一条边 X $\rightarrow$ Y，对于每一个X的剩余可能赋值，Y都有某个赋值方式，使其 不会违反约束，称这条边是一致的

一种简单的传递方式便是，确保所有的边都是一致的
这称为**边一致性算法 arc consistency**

![[AIimg/img1/image-4.png]]
![[AIimg/img1/image-5.png]]

每当 X 失去一个可能的值时，就检查一遍它的所有邻居
当一个变量可取的值为空，说明这种方式不行
这样可以比FC更早发现失败
可以在每一次赋值之前或者之后使用

局限性
- 保证边一致性后，可能无解，却不知道
	![[AIimg/img1/image-6.png|247x107]]
- 边一致性需要在回溯算法的过程中进行计算

Example
https://www.cs.cmu.edu/~15281-s20/demos/csp_backtracking/
# SAT 布尔可满足性问题

通过合适地定义 $x_i$ ，使其取值范围为 $\{True,\ False\}$ 时，CSP问题可以转化为SAT问题，即是否存在一种布尔赋值组合， 使所有的逻辑约束都能被满足

## 结构化的逻辑命题表示

定义
- 常量：True False
- 变量： $x_i$ 取值 True 或 False
- 连接符：与（∧）、或（∨）、非（¬）

任意逻辑关系都可以转为上述三种的组合表达

Example N-皇后

- CSP表述
	![[AIimg/img1/image-7.png|655x200]]

- SAT表述
	![[AIimg/img1/image-8.png]]

关于逻辑表达式，有以下两个结论

摩根定律

$$
\begin{align}
\neg(P \lor Q) \iff (\neg P) \land (\neg Q) \\
\neg(P \land Q) \iff (\neg P) \lor (\neg Q)
\end{align}
$$

分配律
$$
\begin{align}
(P \land (Q \lor R)) \iff ((P \land Q) \lor (P \land R)) \\
(P \lor (Q \land R)) \iff ((P \lor Q) \land (P \lor R))
\end{align}
$$

利用这两个定律，可以将任意逻辑表达式转化为以下两种标准形式

- CNF 合取范式：$(X_1 \lor \neg X_2 \lor X_3) \land (\neg X_3 \lor \neg X_4) \land \dots$
- DNF 析取范式：$(X_1 \land \neg X_2 \land X_3) \lor (\neg X_3 \land \neg X_4) \lor \dots$

其中每个 $X_i$ 或 $\neg X_i$ 称为 **字符**，每个括号的部分称为 **子句**

Example

![[AIimg/img1/image-9.png]]


满足以下约束
1. 每个单元格必须至少填入一个数字 (1-9)：  
   $$
   OR(X(i, j, 1), X(i, j, 2), ..., X(i, j, 9))
   $$

2. 每个单元格最多只能填入一个数字：  
   $$
   NOT(X(i, j, k)) \lor NOT(X(i, j, k'))
   $$
   （对于每个单元格 $(i, j)$ 和每一对不同的数 $k$ 和 $k'$）

3. 每一行中每个数字最多只能出现一次：  
   $$
   NOT(X(i, j, k)) \lor NOT(X(i', j, k))
   $$
   （对于每一对不同的列 $j$ 和 $j'$，和每个固定的数字 $k$）

4. 每一列中每个数字最多只能出现一次：  
   $$
   NOT(X(i, j, k)) \lor NOT(X(i, j', k))
   $$
   （对于每一对不同的行 $i$ 和 $i'$，和每个固定的数字 $k$）

5. 每一个 $3 \times 3$ 的大格每个数字最多只能出现一次：  
   $$
   NOT(X(i, j, k)) \lor NOT(X(i', j', k))
   $$
   （其中 $(i, j)$ 和 $(i', j')$ 在同一个 $3 \times 3$ 子格中）

对于SAT，可以采取更高效的算法

## DPLL 算法

取以下CNF表达式为例 
$$
(p_1 \lor \neg p_3 \lor p_4) \land (\neg p_1 \lor p_2 \lor \neg p_3) \land \dots
$$

对于一个子句，只有以下4种可能
1. Ture：即至少一个字符为真
2. False：即所有字符都为假
3. 单字符：只有一个字符还没有赋值，别的都为假
4. 其他

先考虑使用基本的DFS实现
在DFS中，依次设置每个变量（字符）的值，当其中任意一个子句为假的时候，说明整个表达式为假，需要回溯

在 case3 中，当一个子句只剩一个变量未赋值，而其它均为假时 ，如果将这个变量赋值为假，则该子句为假，需要回溯，所以应该将这个变量设为真

这个过程就是**BCP自动赋值**，而BCP结束后，需要继续设置下一个未赋值的变量，即手动赋值

**单字符传递**：当某个子句只剩下一个字符而其余为假，这个字符应为真
**布尔约束传递 BCP**：重复使用单字符传递，直到无法使用为止

**DPLL = DFS + BCP**

进一步地，如果最后推出矛盾，即整个表达式为False，说明存在一个隐含的约束，可以从矛盾中提取出这个约束，据此，可优化DPLL
## CDCL 算法

**CDCL = DPLL + 从矛盾中得出新的子句**

假设一个表达式 $f$ 由子句 $C_1 ... C_8$ 构成，每个子句内容如图
蓝色代表 F ，橙色代表 T ，灰色代表未赋值
先尝试赋值 $x_1$ ，而后利用手动赋值与BCP，得到如下状态
我们手动赋值了 $x_1$ 和 $x_2$ 为 T，此时可以根据BCP推导

$$
\begin{align}
with (x_1,x_2) = (T, T)\ \ C_5 \implies x_5 = True \\
with (x_1,x_5) = (T, T)\ \ C_2 \implies x_6 = True \\
with (x_5) = (T)\ \ C_3 \implies x_7 = True 
\end{align}
$$
$$
(x_1,x_6,x_7) = (T, T, T) \implies C_4 = False \implies f = False
$$


![[AIimg/img1/image-11.png]]

此时产生矛盾，尝试从中提取隐藏约束
采取一种分割方式，分割线以下是矛盾区域，分割线以上是非矛盾区域
考察穿过分界线的边对应的变量，这是直接导致矛盾的原因
如下的分割方式中， $(x_1 = T,x_5 = T,x_6 = T)$ 导致矛盾，所以得出以下约束
$$
L_1 = (\neg x_1,\neg x_5,\neg x_6)
$$
需要满足 $L_1 = True$
此时，回溯至上一层，考虑 $(x_1,x_5,x_6)$ 但是不能触发BCP

![[AIimg/img1/image-13.png|545x489]]

所以考虑另一种分割

![[AIimg/img1/image-14.png|560x499]]

重复这个过程就行了

Example
https://cse442-17f.github.io/Conflict-Driven-Clause-Learning/




























