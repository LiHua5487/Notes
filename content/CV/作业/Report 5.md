
基本超参数设置如下，在训练过程中，保存 test acc 最高的一个 epoch 的模型

```python
EPOCH=20, BATCH_SIZE=64
```

# Linear Classifier

线性分类器模型结构如下，下面是采用 AdamW 优化器 和 step learning rate schedule 的训练结果

```python
self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, out_channels)
        )
```

![[CV/作业/imgs/img5/image-4.png]]

![[CV/作业/imgs/img5/image-5.png]]

```powershell
(pytorch) PS D:\Files\Project\COURSE_LABS\CV Lab\Problem_Set_5> python main.py --run=test --model=linear --optimizer=adamw --scheduler=step
Files already downloaded and verified
Test Accuracy: 41.24%
```

# FCNN

我的 FCNN 模型结构及相关参数如下

```python
self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.GELU(),
            nn.Dropout(DROP_RATE),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.BatchNorm1d(hidden_channels // 2),
            nn.GELU(),
            nn.Dropout(DROP_RATE),
            nn.Linear(hidden_channels // 2, hidden_channels // 4),
            nn.BatchNorm1d(hidden_channels // 4),
            nn.GELU(),
            nn.Dropout(DROP_RATE),
            nn.Linear(hidden_channels // 4, out_channels)
        )
```

```python
hidden_channels=2048, DROP_RATE=0.2
```

## AdamW vs SGD

相关参数如下，均使用 step learning rate schedule

```python
# SGD
lr=0.01, momentum=0.9
# AdamW
lr=0.001, weight_decay=0.01
```

![[CV/作业/imgs/img5/image.png]]

![[CV/作业/imgs/img5/image-1.png]]

```powershell
(pytorch) PS D:\Files\Project\COURSE_LABS\CV Lab\Problem_Set_5> python main.py --run=test --model=fcnn --optimizer=sgd --scheduler=step  
Files already downloaded and verified
Test Accuracy: 60.03%
(pytorch) PS D:\Files\Project\COURSE_LABS\CV Lab\Problem_Set_5> python main.py --run=test --model=fcnn --optimizer=adamw --scheduler=step
Files already downloaded and verified
Test Accuracy: 60.06%
```


## StepLR vs CosineAnnealingLR

相关参数如下，均使用 AdamW 优化器

```python
# StepLR
step_size=5, gamma=0.25
# CosineAnnealingLR
T_max=20, eta_min=1e-5
```

![[CV/作业/imgs/img5/image-2.png]]

![[CV/作业/imgs/img5/image-3.png]]

```powershell
(pytorch) PS D:\Files\Project\COURSE_LABS\CV Lab\Problem_Set_5> python main.py --run=test --model=fcnn --optimizer=adamw --scheduler=step
Files already downloaded and verified
Test Accuracy: 60.06%
(pytorch) PS D:\Files\Project\COURSE_LABS\CV Lab\Problem_Set_5> python main.py --run=test --model=fcnn --optimizer=adamw --scheduler=cosine
Files already downloaded and verified
Test Accuracy: 59.99%
```




