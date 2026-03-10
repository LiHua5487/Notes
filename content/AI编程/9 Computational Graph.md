
# 基本概念

计算图是表示神经网络运算过程的一种方式，是一个有向无环图 DAG 
- 节点代表操作符
- 边代表数据的流动

操作符包含的范围非常广
- 张量操作：reshape concat matmul
- 网络操作：损失、梯度、优化器
- 数据管理：批处理、预取 Pre-fetch、平铺 Tile、裁剪 Crop、归一化
- 控制流操作：if-else for while

一般来讲，计算图里的边几乎都是**依赖边 dependency**，表示一个节点的运算依赖于另一个节点的输出结果

由于无环，计算图通常是一条直线式的流程，先做 A 、再做 B 、最后做 C ，但现实中的计算可能还有条件判断、循环等逻辑，所以计算图中还存在一种称为 **控制流操作符 control flow** 的特殊操作符，它允许计算图根据数据动态改变执行路径，而不是死板地按顺序运行

# 构建方式

## 动态图

PyTorch的早期版本中，计算图主要是是动态构建的，即随着代码运行逐渐生成

这通常对应着命令式编程，强调“如何做”，代码按步骤执行，每一步都立即计算

```python
import torch
a = torch.tensor(2)
b = torch.tensor(3)
c = a * b + 1
c.backward()
```

- 在这个代码中， `+` `*` 这些操作符都被重载了，执行这个操作时，会顺便将其加入到计算图中
- 在调用 `backward()` 时使用计算图进行反向传播

之所以称之为动态的，是因为在每次迭代中，计算图都是临时构建的，用于前向传播和反向传播，一旦完成这些计算，图就会被销毁，在下一次迭代中重新构建新的图

优点：非常灵活，便于调试，容易支持控制流操作符
缺点：每次迭代都要重新构图，会有开销，且难以优化图

## 静态图

静态计算图则在构建完成后保持不变，直到整个计算任务结束，这种图通常在执行效率方面表现更好，因为可以在执行前对图进行优化

这通常对应着声明式编程：强调“做什么”，先定义整个计算过程，再一次性执行

```python
import tensorflow as tf

a = tf.constant(2)
b = tf.constant(3)
c = a + b

with tf.Session() as sess:
    print(sess.run(c))
```

- 这里也存在操作符的重载，只不过不去执行运算，只是把它放到图里
- 这里构建的实际上是 **Graph IR** (Graph Intermediate Representation，图的中间表示，一种介于编程语言和机器语言中间的东西）
- 而后使用 `Session` 来执行计算，在这个过程中 TensorFlow 会先分析图结构并进行优化

优点：可以更高效，适合图优化和部署
缺点：不够灵活，难以支持控制流操作符

## 动静结合

在 pytorch 中，你可以先写动态图代码（像普通 python 代码一样），然后选择性地将某些函数或模型转换为静态图，这样在训练时可以用动态图快速调试，在推理时用静态图提升速度

将动态图转换为静态图的过程称为图捕获，有两种方法
- 基于跟踪 Trace-based ：提供一个示例输入，记录下执行的操作序列，据此生成一个静态图；但只会记录实际运行到的路径，如果代码中有条件分支，可能无法处理所有情况
- 基于抽象语法树 AST-based / 源码转换：直接分析代码的抽象语法树（即代码的结构），构建一个完整的静态图，支持控制流

下图举例说明了 AST 和计算图的区别

![[AI编程/imgs/img9/image.png]]

```python
import torch
import torch.nn as nn

class SimpleModel(nn.Module):
    def forward(self, x):
        return x * 2 + 1
model = SimpleModel()
# 也可以定义一个函数，并将其转换为静态图

# Trace-based 转换方式
example_input = torch.tensor([1.0])
traced_model = torch.jit.trace(model, example_input)

# AST-based 转换方式
static_model = torch.jit.script(model)

# 此时traced_model和static_model就是静态的了
input_data = torch.tensor([2.0])
output = static_model(input_data)

# 其余部分仍保持动态
c = output ** 2
if c > 100:
	print("too big")
```

---

pytorch 的反向传播有两种模式
- 即时模式 Eagerly ：每次调用 `backward()` 时都动态计算梯度，灵活但可能带来重复开销
- AOT (ahead-of-time) 模式：提前记录前向与反向传播逻辑并生成静态图，然后重用，避免重复工作，效率更高

具体来讲，AOT 自动微分过程如下
- 使用虚拟张量运行前向图，通过自动微分追踪反向过程
- 获取这个联合图，将其分割成一个独立的前向图和后向图
- 将这两个编译后的函数封装成一个 `torch.autograd.Function` 图

```python
import torch
import torch.nn as nn

model = nn.Linear(10, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 使用 torch._dynamo 编译训练步骤，启用 AOT 优化
@torch.compile(backend="aot_eager")
def train_step(x, y):
    optimizer.zero_grad()
    output = model(x)
    loss = nn.functional.mse_loss(output, y)
    loss.backward()  # 反向传播会被 AOT 优化
    optimizer.step()
    return loss

# 运行训练
x = torch.randn(32, 10)
y = torch.randn(32, 1)
loss = train_step(x, y)
```

这里，`torch.compile` 会使用 AOT Autograd 提前分析 `train_step` 的过程（前向和反向传播），生成静态图，在多次训练迭代中，PyTorch 直接运行优化后的代码，而不需要每次都重新构建计算图

# Three Gens of AI Frameworks

- 第一代 Library-based ：允许用户直接调用预定义函数来构建和训练模型，操作灵活，但优化有限
- 第二代 DAG-based ：先定义静态计算图，框架优化后执行，提升性能但牺牲了动态调试的灵活性
- 第三代 AST-based ：通过动态分析代码的抽象语法树生成优化计算图，平衡了编程灵活性和执行效率











