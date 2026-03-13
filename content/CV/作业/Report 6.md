
# Data Augmentation

我实现了两种数据增强方法，分别是随机水平翻转与随机裁剪

```
train_transform = transforms.Compose([
	transforms.RandomHorizontalFlip(),
	transforms.RandomCrop(32, padding=4),
	transforms.ToTensor(),
	transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
```

# Network

对于 VGG 和 ResNet ，训练参数如下，其中 VGG 我少整了俩 block ，因为 CIFAR10 的数据相对简单，模型参数过多会难以训练，且容易过拟合

```
optimizer: AdamW
lr schedule: StepLR
epoch: 20
batch size: 128
```

对于 ResNeXt ，我发现其准确率初始较低且增长缓慢，所以我增加了 epoch 到 50 ，且使用 lr warmup 和 cosine lr schedule 

训练时，我发现中间会出现准确率突然下降然后迅速恢复的情况，这可能是一开始 energy landscape 非常崎岖，且 lr 较大，一不小心就挪到了坏地方，后来 lr 变小且靠近最小点，就比较稳定了

![[CV/作业/imgs/img6/image.png]]

![[CV/作业/imgs/img6/image-1.png]]

![[CV/作业/imgs/img6/image-2.png]]

```powershell
Using model: vgg
Loading checkpoint './checkpoints\vgg_epoch_19.pth'
Loaded checkpoint from epoch 19 with best accuracy 89.37%
Files already downloaded and verified
Test Accuracy: 89.37%

Using model: resnet
Loading checkpoint './checkpoints\resnet_epoch_18.pth'
Loaded checkpoint from epoch 18 with best accuracy 82.56%
Files already downloaded and verified
Test Accuracy: 82.56%

Using model: resnext
Loading checkpoint './checkpoints\resnext_epoch_45.pth'
Loaded checkpoint from epoch 45 with best accuracy 83.24%
Files already downloaded and verified
Test Accuracy: 83.24%
```

---

[模型链接](https://disk.pku.edu.cn/link/AAC8A3C5B04B66456AB1EDFF23BDB1E045)
文件夹名：HW6_model
有效期限：2026-01-02 17:22














