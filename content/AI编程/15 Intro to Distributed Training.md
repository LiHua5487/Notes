
# Distributed System

Distributed ：不只用一个设备（可能是多个电脑，也可能是一个电脑上的多个 GPU ）

一个计算中心里通常长这样，其中一个 node 里通常会有 2 CPU + 8 GPU 
- 一个 node 里，GPU 之间可以通过 NVLink 通信，不再需要 CPU 中转
- 交换机负责不同 node 间的通信

![[AI编程/imgs/img15/image.png]]

不同设备之间传递数据是需要耗时的，而且往往会比计算还要慢

![[AI编程/imgs/img15/image-1.png]]

最理想的情况是，能保证计算是一直进行的，在此期间同时进行数据传输，而不需要某个设备干等着其它设备的数据才能开始计算（这种发生等待的情况称为 stall ）

## Basic Communication Patterns

Push ：设备 A 传递数据给设备 B 
Pull ：设备 A 发出指令给设备 B ，让 B 给自己数据
Scatter ：一个设备的数据分成很多部分，每个部分分别给到一个设备
Gather ：多个设备的数据给到一个设备，存到一起
Broadcast ：一个设备的数据复制若干份，给别的设备
Reduce ：Gather + 合并操作（如求和，取均值）

```
[1] [2] [3] [4] → gather → [1, 2, 3, 4]
[1] [2] [3] [4] → reduce → [1 + 2 + 3 + 4] = [10]
```

All Reduce / Gather ：对所有设备都进行 Reduce / Gather （最后得到一样的结果）

![[AI编程/imgs/img15/image-2.png]]

Wait / Barrier ：一个设备 / 一堆设备需要等待另一个 / 一堆设备计算完了，才能开始计算

![[AI编程/imgs/img15/image-3.png]]

Map Reduce ：先进行 map （把一个操作应用到每个元素上），再进行 reduce ；最大的瓶颈在于 reduce 需要等待 map

# Data Parallelism

## Distributed SGD

每个 epoch 训练时，都需要让整个训练集过一遍模型，但是如果一次性全都塞进去，开销太大了，SGD 的想法是把训练集划分为一堆 batch ，每次只处理一个 batch ，用这个 batch 上的平均梯度来近似整个训练集上的梯度

Distributed SGD 进一步把一个 batch 的数据划分到多个设备上，每个设备只处理一部分数据并计算梯度，最后汇总起来计算整体的平均梯度
- 先把模型 copy 到每个设备上
- 把数据随机均分给每个设备
- 每个设备分别进行前向和反向传播，计算梯度
- 把梯度汇总到一起，计算整体的平均梯度
- 利用这个汇总的梯度更新模型，再重新 copy 给每个设备

![[AI编程/imgs/img15/image-4.png]]

>注意不是每个设备计算梯度后各自更新模型，不然每个设备更新后的结果都不一样，就乱套了

但是这个方法有个缺点，就是通信占比太大了，而且不同步骤是顺序进行的，需要等待
- 第 1 2 4 步是纯通信
- 第 5 步是计算，但是只是拿计算好的梯度更新一下参数，计算太简单了，但是需要更新很多参数，是 memory bound 的
- 只有第 3 步需要比较复杂的计算，是 compute bound 的

![[AI编程/imgs/img15/image-5.png]]

这里白色部分表示空闲状态 idle ，称为 **bubble** 

---

上面的 parameter server 只是一个称呼，不一定非得单独搞一个设备，也可以从那些 worker nodes 里选一个当作主设备，常见的方式有下面几种

![[AI编程/imgs/img15/image-11.png]]

- CPU作为参数服务器：参数存储在 CPU 内存中，GPU 作为工作节点
- 一个GPU作为参数服务器：指定一个 GPU 专门负责参数存储与聚合
- 所有GPU作为参数服务器：即 All-Reduce 模式，各个 GPU 通过 all-reduce 进行通信，汇总梯度信息

>pytorch 的 `torch.nn.DataParallel` 用的是第二种，`torch.nn.parallel.DistributedDataParallel` 用的是第三种

考虑每个设备的 band width （同一时间内要处理的数据量），对于前两种方式，每个 worker 是 $O(1)$ 的，但是 parameter server 是 $O(N)$ 的（$N$ 为设备数，因为需要从各个设备汇总再分发）

![[AI编程/imgs/img15/image-6.png]]

对于第三种方式，直接在这些 worker 间进行 all reduce 操作

![[AI编程/imgs/img15/image-7.png]]

- 如果每次只对一个设备进行 reduce ，时间是 $O(N)$ ，带宽是 $O(N)$ 
- 如果同时进行所有 reduce ，时间缩短到 $O(1)$ ，但带宽增加到 $O(N^2)$ 

常见的设备间通信结构有下面这几种

![[AI编程/imgs/img15/image-8.png]]

## Ring All-Reduce

利用环状结构来实现 all-reduce 操作，为了进一步减小带宽，我们把每个设备上的数据再细分成 $N$ 份（$N$ 为设备数）

先进行 Reduce-Scatter ：按行在环上传递这些数据块并进行累积，直到每个设备都拿到其中一行的完整的数据

![[AI编程/imgs/img15/image-9.png]]

而后进行 All-Gather ：每一行的完整数据按行传递给其它设备，最后每个设备都有所有行的完整数据

![[AI编程/imgs/img15/image-10.png]]

## All-to-All

![[AI编程/imgs/img15/image-12.png]]




































