
# 基本知识

## AI 编译器

编译器：负责把编程语言转换为机器语言

![[AI编程/imgs/img10/image.png]]

- 前端：负责词法分析、语法分析、语义分析，生成中间表示
- 优化：在中间表示上进行各种优化，以提高代码性能或减小体积
- 后端：将优化后的中间表示转换为特定硬件平台的目标代码

LLVM ：一个开源的编译器框架，它把编译器分成多个模块（前端、优化器、后端），这些模块都基于一种通用的中间格式 LLVM IR

![[AI编程/imgs/img10/image-1.png]]

- 因为 LLVM 是开源的，并且模块化，全球的开发者可以共同贡献代码，例如，有人专攻优化部分，有人改进对新型处理器的支持
- 不同的编程语言都可以将自己的代码转换成 LLVM IR，然后直接使用 LLVM 提供的优化器和后端工具

AI 编译器：专门为 AI 模型设计和训练的编译器，接收高级框架（如PyTorch, TensorFlow）定义的模型，通过一系列优化，生成在特定硬件上执行的代码

![[AI编程/imgs/img10/image-2.png]]

- 前端
    - API 转换：将高级框架的API调用转换为内部表示
    - 自动求导：根据前向计算图自动构建反向传播梯度计算图
    - 计算图生成：构建初始的计算图 IR
- 优化
    - 图的优化与调度执行：在计算图层级进行优化，如Batching（批处理）、Cache（缓存）、Overlap（计算与通信重叠）
    - 内核代码优化与编译：在算子层级进行优化，如GPU kernel优化、自动内核生成 (auto kernel generation)
- 后端
    - 硬件特定优化：针对CPU、GPU、AI加速器等不同硬件进行底层代码生成和优化
- 核心目标：辅助编码与调试，优化硬件执行性能 

编译方式
- JIT (Just-In-Time，即时编译) ：在程序运行时动态编译代码，平衡启动速度和长期性能
- AOT (Ahead-of-Time，提前编译)：在程序运行前（如部署时）将代码完全编译成目标平台机器码，以获得最佳运行时性能

IR (Intermediate Representation，中间表示) ：编译器在源代码和目标代码之间使用的、与源语言和目标机器无关的代码表示形式，是编译器优化的核心载体

AST (Abstract Syntax Trees，抽象语法树) ：一种树状结构的IR，表示源代码的抽象语法结构

PASS (优化遍) ：编译器对 IR 进行的一次完整的遍历和处理，通常用于完成一个特定的优化任务

## IR 系统

LLVM IR ：LLVM 提供的与硬件无关的底层 IR ，采用 Linear IR 形式，AI 编译器复用了其成熟的优化器

![[AI编程/imgs/img10/image-3.png]]

Graph IR ：图的中间表示，分为 AST 和 DAG ，比 Linear IR 更适合表达高层次的逻辑（如复杂控制流）

>早期 AI 编译器采用图算分离策略，在 Graph IR 上做图优化，在 LLVM IR 上做算子优化，这种分离的表示阻碍了图与算子之间的联合优化

Mixed IR ：一种结合了图和控制流信息的 IR ，内部常使用 SSA (Static Single Assignment，静态单赋值) 形式，理论上支持图算混合优化，打破图算分离的壁垒，PyTorch就是一个同时拥有 LLVM IR , Graph IR , Mixed IR 的典型例子

多IR系统的优缺点
- 优点：不同 IR 处理不同层级，每种IR可以进行最适合该层级的优化
- 缺点：IR 之间的转换可能引入开销和错误；难以确定跨IR的最佳优化顺序；难以界定某个优化应在哪个 IR 上进行；优化可能改变数值计算顺序，影响模型精度

## AI 编译器前端的工作流程

![[AI编程/imgs/img10/image-4.png]]

- AST Parser ：将源代码解析为 AST ，提取代码的语法结构
- Static Analysis ： 对 AST 进行语法和数据类型检查，验证代码正确性（由 Type System 定义）并构建 IR
- Optimization Pass ： 优化生成的 IR，例如去除冗余计算
- AutoGrad Graph ： 构建梯度计算图，用于支持自动微分和反向传播
- Dispatch and Execution ： 基于图依赖关系，将子图分发到硬件设备上高效执行

# 图的优化遍 Graph Optimization Pass

![[AI编程/imgs/img10/image-5.png]]

## 常量折叠 Constant Fold

如果某个节点的所有输入都是常量，那么在编译时就可以直接计算该节点的输出值，并用一个持有该计算结果的常量节点替换原节点

![[AI编程/imgs/img10/image-6.png]]

Conv-BN Fold ：Batch Norm 层在训练结束后，其参数 $(\gamma, \beta)$ 和统计量 $(\mu, \sigma)$ 是固定的，而前面的卷积层和 BN 都是线性变换，因此可以将两个线性操作合并为一个卷积操作

已知卷积层和 BN 层的计算公式

$$
\begin{align}
z &= W * x + b \\
out &= \gamma \cdot \frac{z - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta
\end{align}
$$

代入即可得折叠后的公式

$$
\begin{align}
out &= W_{fold} * x + b_{fold} \\
W_{fold} &= \gamma \cdot \frac{W}{\sqrt{\sigma^2 + \epsilon}} \\
b_{fold} &= \gamma \cdot \frac{b - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta
\end{align}
$$

## 算子融合 Operator Fusion

把多个小的计算节点打包成一个核函数来执行，减少计算过程中的中间步骤带来的开销（如数据传输、中间结果存储）

融合方法
- 手动融合：使用预先推导并定义好的融合模式，如 Conv-BN-ReLU 融合
- 基于规则的融合：通过预定义的规则模式进行匹配和融合，如XLA
- 基于DOM-Tree的融合：利用图论中的支配树进行融合决策，如TVM

### 手动融合

- 前向传播融合：相对直接，可将 Conv-BN-ReLU 合并
- 反向传播融合：非常复杂，因为梯度计算涉及大量中间结果的依赖，融合能避免将中间结果写回主存，直接在高速缓存中进行计算，极大提升训练效率

![[AI编程/imgs/img10/image-11.png]]

### 基于规则的融合

![[AI编程/imgs/img10/image-10.png]]

常见的融合模式有以下几种
#### Instruction Fusion

把串行的计算打包在一起，比如下面的计算过程

```
temp1 = input * weights
temp2 = temp1 + bias
output = relu(temp2)
```

这里 `temp1` 和 `temp2` 都是中间结果，需要先分配内存并写入，再进行读取，增加了开销，可以把它们放到一个核函数执行，直接计算 

```
output = relu(input * weights + bias)
```

#### Fusion Merger

![[AI编程/imgs/img10/image-7.png]]

对于左图这样的结构，我们将 A 称为生产者，B 和 C 称为消费者，正常需要先存储 A 的结果，然后 B 和 C 进行读取

我们可以把 A 与 B 、 A 与 C 分别打包到两个核函数中，虽然 A 重复计算了，但减少了内存传输的时间开销

#### Sibling Fusion

![[AI编程/imgs/img10/image-8.png]]

对于 Fusion Merger 中的情况，还可以把 B 和 C 打包成一个核函数，减少了一次核函数的调用，且可以利用共享缓存

#### Producer-Consumer Fusion

![[AI编程/imgs/img10/image-9.png]]

还能把 A 和 B 打包在一起，减少核函数的调用次数，而且不用存储 A 到 B 的中间结果，提高内存使用效率

## 布局转换 Layout Transformation

改变张量数据在内存中的排列顺序，以更好地适配不同硬件的内存访问模式（比如使 GPU 的内存访问更连续）和算子的计算需求

常见布局
- NCHW：`[Batch, Channels, Height, Width]` ，通常更适合 GPU 上 Channel 维度的操作
- NHWC：`[Batch, Height, Width, Channels]` 通常对 CPU 更友好，具有更好的访存局部性

![[AI编程/imgs/img10/image-12.png]]

## 内存分配 Memory Allocation

在编译时规划计算图中所有张量所需的内存块，尽可能使用同一块内存而非新的内存（复用），以最小化总内存占用

内存类型
- 静态内存：存储模型参数、常量值、最终输出等，大小固定
- 动态内存：存储中间结果、工作缓冲区等，通常占比较大，是优化的重点

优化方法
- inplace ：如果一块内存不再被需要，且下一个操作是 element-wise 的，可以原地覆盖（比如 B=sigmoid(A) ，如果 A 只被用来算 B ，而 sigmoid 操作是逐元素的，即计算时不依赖其它元素，那就可以在计算出 B 中的一个元素后，直接覆盖掉 A 中的对应位置）
- sharing ：两个数据使用的内存大小相同，且有一个数据参与计算后不再需要，那后一个数据可以覆盖前一个数据

在并行执行路径中，不恰当的内存共享会引入虚假依赖，阻碍并行，优化算法需鼓励在无法并行的节点间进行内存共享

## 代数简化 Algebraic Simplification

利用数学恒等式和算子特性来简化计算表达式，比如
- 合并连续的 reshape ：`Reshape(Reshape(x, shape1), shape2) → Reshape(x, shape2)` 
- 重新结合：`(Mat1 + S1) + (Mat2 + S2) → (Mat1 + Mat2) + (S1 + S2)` 

## 公共子表达式消除 Common Subexpression Elimination

如果同一个表达式在多个地方被计算，可以让该表达式只计算一次，然后将结果复用到所有使用该表达式的地方

算法流程
- 对计算图进行深度优先遍历，得到逆后续序列（即进行拓扑排序）
- 创建一个哈希表用于存储候选子表达式
- 遍历每个节点，计算其哈希值（基于操作类型、输入节点ID等）
- 若哈希值已存在，检查是否真正等价（输入相同，输出类型和数量相同）
- 如果等价，则复用已有节点，删除重复节点

## 死代码消除 Dead Code Elimination

死代码：程序运行过程中，永远不会被执行到的代码，比如 `if(false) {...}` 

计算图中的死代码消除，就是要移除计算图中对最终输出结果没有任何影响的节点，这些无效节点一般来源于其它图优化过程

算法流程
- 获取计算图的逆后续节点集
- 遍历节点，如果一个节点不是计算图的输出节点，且其输出没有被任何其他节点使用，则标记为"死节点"
- 删除死节点，并更新其输入边的连接
- 重复上述过程，直到没有死节点为止

# 图的调度与执行 Graph Dispatch and Execution

- Graph Dispatch ：决定图中每个操作在哪个硬件设备上运行（如 CPU 或 GPU）、以什么顺序执行；会考虑依赖关系，比如某些操作需要等待其他操作的结果，从而避免冲突并最大化并行性
- Graph Execution ：则是实际运行这些操作的过程，按照调度计划计算输出结果

## Graph Dispatch

![[AI编程/imgs/img10/image-13.png]]

在调度时，需要先根据计算图解析依赖关系
- 只有当一个节点的所有依赖输入节点都被计算完成时，该节点才可以被调度运行
- 对于如 `v1` 和 `v2` 这类不依赖其他节点的节点，可以直接进行并行调度执行

我们搞两个队列，并发执行队列用于并行执行计算节点，而调度队列中存放等待执行的节点
- 当执行完并发队列中的一个节点后，就激活其后续节点，并加入到调度队列
- 当并发执行队列存在空位时，从调度队列弹出一个节点补上并执行

## Graph Partition

如果要把一个神经网络分割到多个设备上，可以利用计算图分割来确定如何将一个计算图分配到不同设备上，以及不同设备之间如何通过通信操作 `Send/Recv` （发送 / 接收）进行数据传输

![[AI编程/imgs/img10/image-14.png]]

## Graph Execution

![[AI编程/imgs/img10/image-15.png]]

可以将数据加载、数据处理、训练这些过程异步化执行，在并行计算的基础上进一步提高效率














