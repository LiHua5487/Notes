
# AI frameworks

AI框架是为开发、训练和部署模型提供支持的工具和库，为研究人员和开发人员提供了高效的方式来构建智能算法和应用，通常具备以下功能和组成部分

![[AI编程/imgs/img8/image.png]]

![[AI编程/imgs/img8/image-1.png]]

# Automatic Differentiation

给定一个函数，要计算其微分，如果我们知道这个函数的表达式，可以手动计算出其微分的表达式，再敲到代码里，这就是**手动微分**

更高级一点，可以利用**符号微分**，只需要构建出函数表达式，符号微分就能利用一系列设定好的求导法则（即一个函数/复合函数的导数应该长什么样）给出其微分的表达式但如果函数很复杂，其微分表达式可能很冗长甚至难以表示或计算

**数值微分**不去管微分的具体表达式，而是进行近似，最常见的方法是利用差分近似，但使用数值微分就难免有误差
- 前向差分：`f'(x) ≈ (f(x + h) - f(x)) / h`
- 中心差分：`f'(x) ≈ (f(x + h) - f(x - h)) / (2h)` ，比前向差分更精确，因为利用泰勒展开后能消掉二阶导

**自动微分**将函数分解为一系列基本初等运算（如加、乘、sin、exp等）的序列，按照先后顺序构建出一个计算图，然后应用链式法则，把每一步计算的导数组合起来

![[AI编程/imgs/img8/image-2.png]]

## Forward/Reverse mode AD

在计算微分时，可以采用前向的顺序，即从 $\frac{\partial v_1}{\partial x_1}$ 开始计算并得到关于 $x_1$ 的导数，但是如果还要计算关于 $x_2$ 的导数，就还得跑一遍这个过程

![[AI编程/imgs/img8/image-3.png]]

在神经网络中，一般是输入的变量数很多，输出的变量数较少，如果用前向微分，每个输入变量都得运行一次，太麻烦了，所以一般采用反向微分，从 $\frac{\partial y}{\partial v_k}$ 开始往回算，这里我们给每个节点整一个**伴随值 adjoint** $\bar v_i$ ，即输出结果关于这个节点的导数，这样每个节点直接利用上游节点的伴随值就能计算当前梯度

![[AI编程/imgs/img8/image-4.png]]

在这个过程中，如果一个节点是多个节点的输入，那计算其梯度时就要把每个分支的梯度累加起来，比如上面的 $v_1$ 

$$\overline{v_1} = \frac{\partial y}{\partial v_1} = \frac{\partial y}{\partial v_2} \frac{\partial v_2}{\partial v_1} + \frac{\partial y}{\partial v_3} \frac{\partial v_3}{\partial v_1} = \overline{v_2} \frac{\partial v_2}{\partial v_1} + \overline{v_3} \frac{\partial v_3}{\partial v_1}$$

## Extending Computational Graph

除了上面的反向传播，计算反向微分还可以通过扩展计算图实现，即把每一步求梯度的运算也加入到计算图中

比如有这么一个计算图，反向微分时，结果关于 $v_4$ 的导数是 1 ，可以构建一个右边红色部分这样的节点

![[AI编程/imgs/img8/image-5.png]]

而后看关于 $v_3$ 的导数，由于 $v_4=v_2\cdot v_3$ ，可得

$$\bar v_3=\bar v_4\cdot \frac{\partial v_4}{\partial v_3}=\bar v_4\cdot v_2$$

于是把计算图扩展成这样

![[AI编程/imgs/img8/image-6.png]]

再看 $v_2$ ，其有两个上游节点，先看 $v_2\rightarrow v_4$ 这个分支，记这一部分提供的伴随值为 $\bar v_{2\rightarrow 4}$ ，则

$$\bar v_{2\rightarrow 4}=\bar v_4\cdot \frac{\partial v_4}{\partial v_2}=\bar v_4\cdot v_3$$

再看 $v_2\rightarrow v_3$ 这个分支，由于 $v_3=v_2+1$ ，可得

$$\bar v_{2\rightarrow 3}=\bar v_3\cdot \frac{\partial v_3}{\partial v_2}=\bar v_3\cdot 1=\bar v_3$$

而 $\bar v_2=\bar v_{2\rightarrow 4} + \bar v_{2\rightarrow 3}$ ，于是计算图变成这样

![[AI编程/imgs/img8/image-7.png]]

最后看 $v_1$ ，由于 $v_2=e^{v_1}$ ，可得

$$\bar v_1=\bar v_2\cdot \frac{\partial v_2}{\partial v_1}=\bar v_2\cdot e^{v_1}=\bar v_2 \cdot v_2$$

最终计算图长这样

![[AI编程/imgs/img8/image-8.png]]

那这有啥用呢？在反向传播过程中，计算完梯度，每个节点的伴随值就释放了（如果要储存，开销很大），再次计算时只能从头开始，而计算图在扩展后，天然携带了梯度的计算方式；此外，反向传播难以计算高阶导，而对于扩展计算图，只需要在此基础上再进行扩展，就能计算高阶导

## Reverse mode AD for Tensors

上述过程都只涉及标量，但还可能涉及矩阵，比如以下计算图

![[AI编程/imgs/img8/image-9.png]]

对于矩阵，其伴随值的定义如下

$$\bar Z = \begin{bmatrix} \frac{\partial y}{\partial Z_{1,1}} & \cdots & \frac{\partial y}{\partial Z_{1,n}} \\ \vdots & \ddots & \vdots \\ \frac{\partial y}{\partial Z_{m,1}} & \cdots & \frac{\partial y}{\partial Z_{m,n}} \end{bmatrix}$$

在计算 $X$ 的伴随值时，由于需要计算矩阵关于矩阵的导数，我们拎出来 $X$ 中的一个元素来看，由于 $Z_{ij}=\sum_k X_{ik}\cdot W_{kj}$ ，则 $X_{ik}$ 的伴随值为

$$\overline{X_{i,k}} = \sum_j \frac{\partial Z_{i,j}}{\partial X_{i,k}}\cdot \overline{Z_{i,j}} = \sum_j W_{k,j} \cdot \overline{Z_{i,j}}$$

综合成矩阵形式，就是

$$\overline{X} = \overline{Z} W^T$$

而对于 $W$ ，就是

$$\overline W=X^T\overline Z$$

## Reverse mode AD for Structures

在计算图中，还可能出现一些特殊的运算，比如取最大值、条件判断、根据键查找字典等，在反向微分时，我们要考察具体是哪些值真正参与了后续计算，以查字典为例，其伴随值的定义如下

![[AI编程/imgs/img8/image-10.png]]

由于我们使用键 cat 进行查找，所以 $b=a_0$ ，即 $a_0$ 参与了后续计算，那反向微分时，就应该计算关于 $a_0$ 的梯度，而 $\frac{\partial b}{\partial a_0}=1$ ，所以使用键 cat 的情况下，$d$ 的伴随值就是 $\bar b$ ，于是将其伴随值也设成一个字典，键 cat 对应的值就是 $\bar b$  

## Jacobian Matrix and Comparison

注意到，在定义矩阵的伴随值时，用的正好是**雅可比矩阵 Jacobian Matrix**，对于一个函数 $f:R^n\rightarrow R^m$，其可以写成

$$f(x) = \begin{pmatrix} f_1(x_1, x_2, \ldots, x_n) \\ f_2(x_1, x_2, \ldots, x_n) \\ \vdots \\ f_m(x_1, x_2, \ldots, x_n) \end{pmatrix}$$

其雅可比矩阵定义如下

$$J(x) = \frac{\partial (f_1\cdots f_m)}{\partial (x_1 \cdots x_n)} = \begin{pmatrix} \frac{\partial f_1}{\partial x_1} & \frac{\partial f_1}{\partial x_2} & \cdots & \frac{\partial f_1}{\partial x_n} \\ \frac{\partial f_2}{\partial x_1} & \frac{\partial f_2}{\partial x_2} & \cdots & \frac{\partial f_2}{\partial x_n} \\ \vdots & \vdots & \ddots & \vdots \\ \frac{\partial f_m}{\partial x_1} & \frac{\partial f_m}{\partial x_2} & \cdots & \frac{\partial f_m}{\partial x_n} \end{pmatrix}_{m\times n}$$

而对于标量关于矩阵的导数，长这样

$$\frac{\partial f}{\partial A} = \begin{pmatrix} \frac{\partial f}{\partial A_{11}} & \frac{\partial f}{\partial A_{12}} & \cdots & \frac{\partial f}{\partial A_{1n}} \\ \frac{\partial f}{\partial A_{21}} & \frac{\partial f}{\partial A_{22}} & \cdots & \frac{\partial f}{\partial A_{2n}} \\ \vdots & \vdots & \ddots & \vdots \\ \frac{\partial f}{\partial A_{m1}} & \frac{\partial f}{\partial A_{m2}} & \cdots & \frac{\partial f}{\partial A_{mn}} \end{pmatrix}$$

矩阵关于矩阵的导数，就是矩阵中每个元素关于矩阵的导数排列到一起，是一个高维张量

从上面的微分计算中可知，反向微分中一般用的是雅可比矩阵的转置 $J^T\cdot \bar v$ ；而前向微分一般是 $J\cdot \dot v$ ，所以如果输入变量更多，适合用前向微分；输出变量更多，适合用反向微分









