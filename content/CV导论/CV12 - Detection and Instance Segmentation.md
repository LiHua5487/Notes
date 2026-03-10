---
date: 2025-05-21
tags:
  - CV
---
# Object Detection


![[CV导论/imgs/img12/image.png]]

## Single Object

只有一个主要物体，只用输出一个**包围框 bounding box (b box)**
tight bounding box 包含整个物体，且不能更小

2D 一般要求 axis align 框边缘与坐标轴平行，3D 一般不要求
如果不要求 axis align ，得到的框可能更贴合（比如斜着的矩形框）

可以用 4 个变量描述一个框：$x, y$ 框左上角坐标， $h,w$ 框高宽

Localization + Classification
![[CV导论/imgs/img12/image-1.png]]

定位时，如果不加限制，输出的范围可能超过允许范围
可以先 sigmoid ，再乘以范围允许的最大值

分类可以用 CE Loss ，而对于定位，以下Loss
- L1 loss $\Sigma |\Delta_i|$ ：梯度大小固定，有一定鲁棒性，但对于靠近最小值的地方表现不好，不利于达到最小值
- L2 loss $\Sigma \Delta_i^2$  ：梯度正比于 error ，利于梯度下降，但对于很大的 error ，就有很大的梯度，表现不好
- Rooted mean squared loss (RMSE) $\sqrt{\frac{1}{N} \Sigma \Delta_i^2}$  ：由于开根号，靠近 0 的地方梯度会非常大

注：L1 Loss = L1 Norm ，但 L2 Loss ≠ L2 Norm (sqrt L2 Loss)

可以将 L1 与 L2 结合，称为 smooth L1 Loss

![[CV导论/imgs/img12/image-2.png]]

## Multi Object

不知道要检测出多少个物体，而且物体大小差异可能很大

一个简单的想法，在图片上利用 sliding window 截取区域，每个区域进行分类检测，判断能不能作为边界框，尝试不同位置不同大小
但这显然开销太大了

## R-CNN

从图片中取出小区域 region proposals ，利用插值等方法将其处理成标准大小，对这些小区域用 single object detection

![[CV导论/imgs/img12/image-3.png]]

对于每个小区域，一方面通过 SVM 进行分类，另一方面得到 $(dx,dy,dw,dh)$ ，指示边界框应该怎么变化更贴合 correction
对于缩小，可以理解，但是对于扩大，这个区域都不知道外部信息，那咋知道扩多少，只能猜测

而且，小区域可能很多，处理速度很慢

## Fast R-CNN

考虑不直接从图片截取区域，而是从处理后的特征图上截取
因为每个cell的感受野更大，可以一定程度上感知到周围信息

![[CV导论/imgs/img12/image-4.png]]

对于原始图像上应用的 region proposal ，可以通过按比例缩放对应到特征图的相应位置，如果不是正好卡在网格上，需要进行“吸附”

![[CV导论/imgs/img12/image 1.png]]

每个区域可以再分成一堆 2×2 的部分，分别进行池化，称为 RoI Pool
得到这个区域的特征，统一维度大小后再进行处理

由于截取出的特征更小，处理速度比 R-CNN 快很多
这时 region proposal 就成了主要耗时部分

## Faster R-CNN

把 region proposal 也用神经网络处理，称 RPN，其余部分与 Fast R-CNN 相同

![[CV导论/imgs/img12/image-5.png|456x380]]

先在特征图上利用 sliding window （此称 anchor box） 得到区域截取
相比于直接在原始图像用 sliding window ，特征图维数更小
而且，由于感受野提高，只用得到更少的区域就能涵盖全图

每个 anchor box 先进行分类检测，如果存在目标物体，就进一步根据 ground-truth 得到 correction
当分类检测结果为 False 时，训练时就不管定位的 Loss 了（置零）

分类检测中的概率称为 objectness ，根据这个概率排序，取前面一些个

实际使用时，每个位置用 K 个 anchor box 尝试，对应不同的长宽比与大小

整体来讲，分为两个阶段

![[CV导论/imgs/img12/image-6.png]]

第一个分类是二分类，判断有没有物体；第二个分类判断具体类别
YOLO 把这两个阶段合并为一个，最后进行 N+1 分类，速度更快，但检测质量可能较差

## NMS

对于一个物体，可能有多个候选的 bounding box 
每个框对应一个 confidence score （Faster RCNN 中的分类评分）

对于N-分类，其中最大的概率对应分类结果，这个概率称 top 1 probability ，作为边界框的置信度  confidence score 

但是最后得到的一堆框并不知道各自对应哪个物体，想从中选取出一些框，分别框住不同的物体，并去掉多余的

设置一个 IoU 的阈值，代表两个边界框对应的是不是一个物体  

先选取出所有框里 confidence score 最大的一个
将剩下的框与之对比，高于 IoU 阈值的，代表可能对应同一个物体，产生重复，就扔了，低于的保留

从保留下来的框里再选出 confidence score 最大的一个，并重复上述过程

## Evaluation Metric

评价标准
- Good accuracy (precision) ：减少错误检测
- Good localization (precision) ：最大化 IoU
- Single response constraint (precision) ：去掉多余的检测结果
- Good coverage (recall) ：想让所有的目标都被检测到了

$$
\text{Precision} = \frac{\text{True Positives (TP)}}{\text{True Positives (TP)} + \text{False Positives (FP)}}
$$

$$
\text{Recall} = \frac{\text{True Positives (TP)}}{\text{Total Ground-Truth Boxes (GT)}}
$$
- 按照置信度顺序从大到小排列，依次计算考虑到每一个框时累计的 Precision 与 Recall
- 如果一个预测框的IOU ≥ 阈值，且它匹配的 Ground-Truth 框尚未被匹配，则该框为 TP
- 如果 IOU < 阈值 或者预测框多次匹配同一个 Ground-Truth 框，则为 FP

AP (Average Precision)
- 每个类别的物体都有一堆边界框，按照 confidence score 由大到小排序
- 选取前 n 个计算 recall
- 计算对应的 precision ，即 IoU 超过一定阈值的边界框的比例
- 计算 precision-recall 曲线围成的面积（AP）

![[CV导论/imgs/img12/image-7.png|465x335]]

还可以对于所有 IoU 的 AP 取平均，而这是对于一个事物类别的，所有类别再取平均，就得到了 mAP

# Instance Segmentation

一种思路是，先得出边界框，每个框内进行 segmentation (top-down)
另一个思路是先得到 mask ，再判断不同像素是不是一类 (bottom-up)

对于2D情况，top-down 一般会更好

top-down 最后分类时，实际上进行 C 分类，根据前面的分类评分选择

## Mask-RCNN

![[CV导论/imgs/img12/image-8.png]]

## RoI Align

如果直接根据对 Pool 后 2×2 的特征图还原边界框，精度会很差
所以考虑弄到 14×14

此外，RoI Pool 中吸附会导致还原回边界框时信息丢失
考虑不进行吸附，对于不在网格上的点，根据附近4个点进行双线性插值

![[CV导论/imgs/img12/image-9.png]]








