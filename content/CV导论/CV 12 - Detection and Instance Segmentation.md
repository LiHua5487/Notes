---
date: 2025-05-21
tags:
  - CV
---

# Object Detection

**目标检测 object detection**既要判断图里有哪些物体，也要给出它们的位置，通常为每个物体输出一个类别、一个边界框和一个置信度

![[CV导论/imgs/img12/image.png]]

## Single Object

如果只有一个主要物体，定位部分只需输出一个**边界框 bounding box（bbox）**，分类部分再判断它是什么
tight bounding box 包含整个物体，且不能更小

2D 一般要求 axis aligned 框边缘与坐标轴平行，3D 一般不要求
如果不要求 axis aligned ，得到的框可能更贴合（比如斜着的矩形框）

可以用 4 个变量描述一个框：$x, y$ 框左上角坐标， $h,w$ 框高宽

Localization + Classification
![[CV导论/imgs/img12/image-1.png]]

定位时，如果不加限制，输出的范围可能超过允许范围
可以先 sigmoid ，再乘以范围允许的最大值

分类可以用 CE Loss ，而对于定位，以下Loss

- L1 loss $\Sigma |\Delta_i|$ ：梯度大小固定，有一定鲁棒性，但对于靠近最小值的地方表现不好，不利于达到最小值
- L2 loss $\Sigma \Delta_i^2$  ：梯度正比于 error ，利于梯度下降，但对于很大的 error ，就有很大的梯度，表现不好
- 均方根误差 RMSE $\sqrt{\frac1N\sum_i\Delta_i^2}$：对残差向量的 L2 范数进行缩放，不能只看外层平方根就认为靠近 0 时梯度一定爆炸，链式法则中还会乘上残差，零点处则需要处理可导性或数值稳定性

注：L1 Loss = L1 Norm ，但 L2 Loss ≠ L2 Norm (sqrt L2 Loss)

可以将二者结合，得到 **smooth L1 loss**：误差较小时采用二次形式，靠近正确位置时平滑优化；误差较大时采用线性形式，减弱离群误差的影响，边界框回归就是学习连续坐标或坐标修正量

![[CV导论/imgs/img12/image-2.png]]

## Multi Object

不知道要检测出多少个物体，而且物体大小差异可能很大

一个简单的想法，在图片上利用 sliding window 截取区域，每个区域进行分类检测，判断能不能作为边界框，尝试不同位置不同大小
但这显然开销太大了

## R-CNN

从图片中取出小区域 region proposals ，利用插值等方法将其处理成标准大小，对这些小区域用 single object detection

![[CV导论/imgs/img12/image-3.png]]

原始 R-CNN 对候选区域提取 CNN 特征，再通过 SVM 分类，并用边界框回归预测位置和大小的修正量 $(d_x,d_y,d_w,d_h)$，让候选框更贴合目标
即使框没有包含整个目标，模型也可能根据看到的局部特征学到应当如何修正，但外部信息不足仍会限制定位能力

而且，小区域可能很多，处理速度很慢

## Fast R-CNN

考虑不直接从图片截取区域，而是从处理后的特征图上截取
因为每个cell的感受野更大，可以一定程度上感知到周围信息

![[CV导论/imgs/img12/image-4.png]]

对于原始图像上应用的 region proposal ，可以通过按比例缩放对应到特征图的相应位置，如果不是正好卡在网格上，需要进行“吸附”

![[CV导论/imgs/img12/image 1.png]]

候选区域称为 **RoI (region of interest)**，**RoI pooling**把每个大小不同的 RoI 划成固定数量的网格，比如 $2\times2$，对每格分别池化，得到固定尺寸的特征，再进行分类与边界框回归，RoI 指整个候选区域，而不是其中的小格

由于截取出的特征更小，处理速度比 R-CNN 快很多
这时 region proposal 就成了主要耗时部分

## Faster R-CNN

把产生候选区域的环节也换成神经网络，称为 **region proposal network (RPN)**，与后续检测部分共享图像特征，减少单独生成 proposal 的开销

![[CV导论/imgs/img12/image-5.png|456x380]]

RPN 在特征图的各个位置滑动处理，每个位置对应原图中的若干个参考框，称为 **anchor box**，这些框具有预设的大小和长宽比，滑动窗口是计算方式，anchor 则是用来预测目标位置的参考框

RPN 为每个 anchor 预测前景/背景分数，以及边界框修正量，训练时根据 anchor 与真实框的匹配关系分配正负标签，通常只对正样本计算定位损失；推理时则完全依靠模型预测，不再使用 ground truth

分类检测中的概率称为 objectness ，根据这个概率排序，取前面一些个

实际使用时，每个位置用 K 个 anchor box 尝试，对应不同的长宽比与大小

整体来讲，分为两个阶段

![[CV导论/imgs/img12/image-6.png]]

第一个分类是二分类，判断有没有物体；第二个分类判断具体类别
YOLO 等单阶段检测器直接在特征上预测目标位置与类别，省去单独的候选区域检测阶段，具体分类和置信度定义随版本变化，速度与精度也需要结合具体模型比较

## NMS

对于一个物体，可能有多个候选的 bounding box
每个框对应一个 confidence score （Faster RCNN 中的分类评分）

在常见检测头中，每个候选框会有类别分数，可以用相应类别的分数排序，具体是否还结合 objectness 取决于模型，不能把所有检测器的置信度都等同为最大 softmax 概率

但是最后得到的一堆框并不知道各自对应哪个物体，想从中选取出一些框，分别框住不同的物体，并去掉多余的

通常先按类别分组，再设置 IoU 阈值，用框的重叠程度近似判断是否在重复检测同一个目标，迭代进行以下步骤

- 先选取出所有框里 confidence score 最大的一个
- 将剩下的框与之对比，高于 IoU 阈值的，代表可能对应同一个物体，产生重复，就扔了，低于的保留
- 从保留下来的框里再选出 confidence score 最大的一个，并重复上述过程

## Evaluation Metric

评价标准

- 准确性：减少错误类别或没有对应目标的检测
- 定位精度：预测框与真实框有足够高的 IoU
- 单一响应：同一个目标尽量只检测一次
- 覆盖程度：尽量找全真实目标，减少漏检

$$
\text{Precision} = \frac{\text{True Positives (TP)}}{\text{True Positives (TP)} + \text{False Positives (FP)}}
$$

$$
\text{Recall} = \frac{\text{True Positives (TP)}}{\text{Total Ground-Truth Boxes (GT)}}
$$

- 按照置信度顺序从大到小排列，依次计算考虑到每一个框时累计的 Precision 与 Recall
- 在类别正确的前提下，如果预测框与真实框的 IoU 达到阈值，且这个真实框尚未被更高分预测匹配，就计为 TP
- 类别错误、没有达到匹配阈值，或者重复匹配已被占用的真实框，通常计为 FP，具体评测还要遵循数据集的忽略区域等约定

**平均精度 AP (average precision)**将不同置信度阈值下的 precision、recall 综合起来

1. 对一个类别，将所有预测按置信度从高到低排序
2. 依次加入预测，按匹配规则累计 TP、FP，计算相应的 precision、recall
3. 得到 PR 曲线，再按评测规定进行插值、积分或离散采样，得到 AP

只保留高分框时，通常误检较少，但会漏掉低分真目标；逐渐放宽阈值，召回率上升，也可能引入更多误检，AP 用于综合这段变化

![[CV导论/imgs/img12/image-7.png|465x335]]

对不同类别的 AP 取平均，得到 **mAP**，有的指标固定匹配阈值，如 AP@0.5，有的还在一组指定 IoU 阈值上取平均，因此比较数值时需要确认采用同一套评测规则

# Instance Segmentation

一种思路是，先得出边界框，每个框内进行 segmentation (top-down)
另一个思路是先预测像素级信息，再把属于同一个实例的像素聚合起来 (bottom-up)，这里要区分“同一语义类别”和“同一个具体物体”

top-down 可以复用目标检测器，下面的 Mask R-CNN 就采用这种思路

一些 top-down 方法会为每个候选框预测 $C$ 个类别对应的 mask，再根据检测出的类别选择相应 mask

## Mask R-CNN

在 Faster R-CNN 的分类、边界框回归之外，再增加一个 mask 分支，为每个 RoI 预测像素级前景区域

![[CV导论/imgs/img12/image-8.png]]

## RoIAlign

分割需要保留更精细的空间位置，通常需要比示意图中 $2\times2$ 更高的 RoI 特征分辨率，比如 $14\times14$

RoI pooling 还会把区域边界和分格位置取整，导致原图与特征之间出现错位，**RoIAlign**保留浮点坐标，在每格中选取采样位置，通过附近 4 个网格点的双线性插值取值，再进行聚合，从而更准确地对齐区域特征，帮助恢复目标的 mask

![[CV导论/imgs/img12/image-9.png]]
