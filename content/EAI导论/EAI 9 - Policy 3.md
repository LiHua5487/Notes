
关键词：**Generalized Advantage Estimation (GAE)**

---
# Critics as Baselines

为了减小 policy gradient 的方差，上回中说到了三种操作
- 将全程的累计奖励替换为 reward to go
- 又将 reward to go 替换为 Q 值
- 给 Q 值减一个 baseline 值函数
又引入了神经网络估计值函数，最后得到了 Actor-Critic 算法

$$
\nabla_\theta J(\theta) \approx \frac{1}{N} \sum_{i=1}^N \sum_{t=1}^T \nabla_\theta \log \pi_\theta (\mathbf{a}_{i,t} | \mathbf{s}_{i,t}) \left( r(\mathbf{s}_{i,t}, \mathbf{a}_{i,t}) + \gamma \hat{V}^\pi_\phi (\mathbf{s}_{i,t+1}) - \hat{V}^\pi_\phi (\mathbf{s}_{i,t}) \right)
$$

- 由于 critic 的存在，其方差较小
- 但 critic 估计不一定准确，所以不是无偏的，和真值不一定相同

而 policy gradient 如下

$$
\nabla_\theta J(\theta) \approx \frac{1}{N} \sum_{i=1}^N \sum_{t=1}^T \nabla_\theta \log \pi_\theta (\mathbf{a}_{i,t} | \mathbf{s}_{i,t}) \left( \left( \sum_{t'=t}^T \gamma^{t'-t} r(\mathbf{s}_{i,t'}, \mathbf{a}_{i,t'}) \right) - b \right)
$$

- 由于只采样了一个轨迹，方差较大
- 由上回的推导可得，减去常数项仍是无偏的

有没有什么办法能折中一下呢，可以考虑在一条轨迹中只往下看 n 步，后面的情况采用值函数去代表，而这个值函数使用神经网络估计

![[EAI导论/imgs/img9/image.png]]

这种情况下，优势函数如下，通过调整 n ，在减小方差和无偏之间进行 trad-off ，一般 n 不取太大（如 2 或 4）

$$
\hat{A}^\pi_n(\mathbf{s}_t, \mathbf{a}_t) = \sum_{t'=t}^{t+n} \gamma^{t'-t} r(\mathbf{s}_{t'}, \mathbf{a}_{t'}) + \gamma^n \hat{V}^\pi_\phi(\mathbf{s}_{t+n}) - \hat{V}^\pi_\phi(\mathbf{s}_t)
$$

- $n = 1$ 时变为 Actor-Critic 
- $n \rightarrow \infty$ 时变为带 bias 的 policy gradient 

但 n 可以不只取一个值，可以让 n 取遍所有整数值，然后进行加权求和

$$
\hat{A}^\pi_{\text{GAE}}(\mathbf{s}_t, \mathbf{a}_t) = \sum_{n=1}^\infty w_n \hat{A}^\pi_n(\mathbf{s}_t, \mathbf{a}_t)
$$

为了得到递归形式，取权重 $w_n \propto \lambda^{n-1}$ ，其中 $\lambda \in (0,1)$ ，化简可得

$$
\begin{aligned}
&\hat{A}^\pi_{\text{GAE}}(\mathbf{s}_t, \mathbf{a}_t) = \sum_{t'=t}^\infty (\gamma \lambda)^{t'-t} \delta_{t'} \\
&\text{where} \quad \delta_{t'} = r(\mathbf{s}_{t'}, \mathbf{a}_{t'}) + \gamma \hat{V}^\pi_\phi(\mathbf{s}_{t'+1}) - \hat{V}^\pi_\phi(\mathbf{s}_{t'})
\end{aligned}
$$

可以发现这正是上回中 Option1 的形式，其中 $\gamma \lambda$ 作为新的折扣因子，但和只用 $\gamma$ 效果上是一样的，最后得到的这个式子就叫做 **Generalized Advantage Estimation (GAE)**









