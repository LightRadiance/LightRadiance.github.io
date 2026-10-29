---
layout: post
title: Rethink the Effectiveness of PER under GPI 
date: 2026-10-29 
description: 在一个 50 状态的稀疏奖励链上，用精确 MSVE 检验 PER 是否真的加速收敛。结论是否定的：PER+IS 没有稳定加速，去掉 IS 才快但不稳定；我提出的逆频率采样在表格中有效，在神经网络下失效。
tags: [强化学习, 经验回放, PER]
---

PER 自 2017 年被提出以来产生了诸多变种，然而却始终难以成为 RL 中的默认组件，不得不怀疑: **PER 真的有助于策略迭代吗?**

PER 背后的直觉很简单：优先回放 TD 误差大的样本，价值函数会收敛得更快。这个叙事在学术中被反复使用，而证明往往是在策略评估下进行，即在策略固定不变的条件下， PER 会加速状态价值估计的收敛，但是鲜有有实验去验证在 **GPI 框架下, 这种优越性是否存在。** 

我想把这个假设单独拿出来，在一个没有神经网络、没有函数逼近误差的环境里检验。如果 PER 的加速来自“更好的样本顺序”，那么在最干净的表格环境里进行策略迭代，它应该很明显。

这篇文章总结我在 PER-eval 项目里的一组实验。在这个受控设定下，我没有观察到 PER 对 MSVE 收敛的加速；而且 PER 的加速与重要性采样（IS）之间存在一个结构性矛盾：去掉 IS 快但不稳定。我据此尝试了逆频率采样, 虽然在表格形式上有效，但引入 NN 后同样收益甚微。

## 1. 问题：PER 的加速从哪里来？

PER 给每个转移一个优先级 $p_i$，采样概率正比于优先级，再用重要性采样权重 $w_i$ 修正非均匀采样带来的偏差。

$$
p_i = (\lvert \mathrm{TD}_i \rvert + \varepsilon)^\alpha
$$

但 TD 误差是一个局部量。它只说明“当前这一步的估计与自举目标不一致”，并不说明“这次更新对全局 MSVE 的贡献有多大”。
接下来试图通过实验回答以下三个问题：

1. **在表格链上，以 MSVE 为评价指标，PER + IS 是否优于均匀回放？**
2. **如果去掉 IS，加速是否出现？代价是什么？**
3. **如果 PER 不适合这个任务，是否存在更简单的替代？**

## 2. 实验设置

主实验是一条 50 状态的确定性马尔可夫链：从 $s_1$ 出发，动作只有向左、向右，只有走到 $s_{50}$ 才得到 $+1$ 的奖励，其余转移奖励为 0；折扣因子 $\gamma = 0.99$，单回合上限 500 步。值函数用表格 Q-learning 学习，学习率默认 0.1，$\varepsilon$ 从 1.0 线性衰减到 0.05。

| 项目 | 设置 |
| --- | --- |
| 环境 | MarkovChain-v0，50 状态，稀疏奖励，确定性转移 |
| 算法 | 表格 Q-learning，$\gamma = 0.99$，buffer size = 100k |
| 采样 | uniform / PER（$\alpha = 0.6$，$\beta: 0.4 \to 1$）/ PER 无 IS（$\beta = 0$）|
| 种子 | 10 个(0-9) |
| 预算 | 60k 环境步，每 1k 步评估一次 |
| 指标 | MSVE：$V(s) = \max_a Q(s,a)$ 与真实 $V^*(s)$ 在 49 个非终止状态上的均方误差 |

选择表格链的原因在于干净：没有函数逼近误差，没有网络初始化，没有泛化；Q 表的每个状态是独立的参数，任何性能差异只能来自采样顺序和更新幅度。MSVE 可以直接对真实值计算，不需要估计。

在稀疏奖励下，“首次发现奖励”的时间本身有很大的种子方差，而它和“发现之后价值传播多快”是两件事。所有绝对收敛步数都混合了这两个成分。后文会用一个粗略的对齐来分离它们，但这个混杂不能被完全消除。

## 3. PER：没有稳定的加速

<figure>
  <img src="{{ '/assets/images/per-reeval/01_fixed_uniform_per_msve.png' | relative_url }}" alt="4个 batch 设置下 Uniform 与 PER 的 MSVE 曲线">
  <figcaption>图 1: 固定学习率, 不同 batch 下, uniform 与 PER 的 MSVE 曲线</figcaption>
</figure>

图 1 是主结果，4个 panel 分别是 batch 1、8、32、64，最直接的观察是：在 GPI 框架下，PER 在任何一个设置里都没有明显加速。

<figure>
  <img src="{{ '/assets/images/per-reeval/02_tuned_uniform_per_msve.png' | relative_url }}" alt="4个 batch 设置下 Uniform 与 PER 的 MSVE 曲线">
  <figcaption>图 2: 调优学习率, 不同 batch 下, Uniform 与 PER 的 MSVE 曲线</figcaption>
</figure>

为了排除学习率的干扰，在不同 batch 下，我按照 coarse search + fine tuning 的方式进行了调优，对结论没有影响。

## 4. 矛盾：IS 的阻尼

标准 PER 的 IS 权重是

$$
P(i) = \frac{p_i}{\sum{p_j}}
$$

$$
w_i = (\frac{\frac{1}{N}}{P(i)})^{\beta}
$$

$$
w_i = \frac{w_i}{\max{w_j}}
$$

其中 $P(i)$ 是转移 $i$ 被采样的概率，$N$ 是回放池大小。这个归一化的目的是让最大的权重等于 1，从而保证更新幅度不超过名义学习率。在 $\beta = 1$ 时，经过简单推导, 它可以写成更直观的形式：

$$
w_i = \frac{p_{\min}}{p_i}
$$

优先级越高，权重越小。一个 TD 误差很大的样本，恰恰是 PER 最想回放的样本，也是权重被压得最小的样本。

这就形成了一个矛盾的局面：PER 用更高的采样频率来补偿，但采样频率的提升受 batch 大小约束，而权重阻尼是乘性的、持续的。两者相抵，净更新量反而更小。$\beta$ 从 0.4 退火到 1 的过程，也是阻尼逐渐加深的过程；而价值传播最活跃的阶段正好落在这段退火区间里，步长被持续压低。

这里需要澄清一点：IS 保证的是“期望更新方向”无偏，不是“更新幅度”不变。最大权重归一化把所有样本的权重压到 1 以下，在平均意义上把整个更新的尺度压小了；固定点不变，但到达固定点的速度变慢。

这种矛盾性进一步限制了 PER 潜在的加速机制，那如果把 IS 去掉呢？ 


## 5. 去掉 IS：加速出现了，但代价是稳定性

<figure>
  <img src="{{ '/assets/images/per-reeval/03_fixed_uniform_per_nois_msve.png' | relative_url }}" alt="去掉 IS 后 PER 在 不同batch 上的 MSVE 曲线">
  <figcaption>图 3: 固定学习率, 不同 batch 下, uniform、PER 与 PER(no IS) 的 MSVE 曲线</figcaption>
</figure>

如果 IS 是阻尼的来源，那么把 $\beta$ 设为 0、完全去掉 IS，应该能看到 PER 的“原始速度”。实验结果确实如此，但只在很窄的窗口里成立，由于没有 IS 的权重压制，在 Batch = 64 的设置下甚至出现了发散。

那如果直接调节学习率呢?

<figure>
  <img src="{{ '/assets/images/per-reeval/04_tuned_uniform_per_nois_msve.png' | relative_url }}" alt="去掉 IS 后 PER 在 不同batch 上的 MSVE 曲线">
  <figcaption>图 4: 调优学习率, 不同 batch 下, uniform、PER 与 PER(no IS) 的 MSVE 曲线</figcaption>
</figure>

除 Batch = 1 的设置外，可以看到，直接去掉 IS 与调节学习率对 MSVE 收敛所产生的影响是一致的。

整体分析来看，无论是 PER，还是 PER(no IS)，对 MSVE 的收敛速度都没有显著且一致的加速。


## 6. 为什么：TD 误差是局部的

把上面的现象放在一起，可以尝试进行分析。

TD 误差度量的是“这一步的自举目标与当前估计差多少”。它是一个局部量，不知道这次更新对全局 MSVE 的贡献。EVB（expected value of backup）的概念来自 Mattar 与 Daw 对规划中记忆回放的规范性解释。Li 等人在 2021 年的工作里进一步证明：在 Q-learning 中，TD 误差的绝对值给出了 EVB 的一个上界（差一个步长因子）；在他们的线性网格例子里，按 TD 优先级需要 $4N$ 次回放，按 EVB 只需要 $N$ 次。

真正能加速全局传播的，是“这次 backup 能让多少状态的值向前推进”。Prioritized sweeping（Moore & Atkeson, 1993）用模型和优先队列来做这件事：它知道状态之间的前驱关系，可以把更新直接推向波前。

我们的 no-IS 结果和这个区分是一致的：去掉 IS 后，PER 在“发现之后的传播阶段”确实更快，但它没有把这种局部优势转化为端到端的加速——发现更晚、且更容易不稳定。PER + IS 则连局部优势都被阻尼抵消了。

在上面实验中，端到端变慢的直接原因是 IS 阻尼；TD 的局部性则解释了为什么 PER 的优先级不能转化为全局加速。

## 7. 一个替代尝试：逆频率采样

Uniform 对所有 transition 进行均匀采样，导致少量有价值 transition 的学习迟滞; PER 偏向有价值的 transition, 但过分关心局部无法提升全局的收敛性。考虑到收敛定理要求的所有状态无限次访问条件, 一个直觉是, **如果我尽可能均匀的访问每一个状态, 全局的收敛性是不是会加快呢？** 

具体做法很简单：对回放池里的每个状态 $s$，记它在池子里的转移数为 $B(s)$；采样时先在所有出现过的状态上均匀抽一个状态，再在这个状态的转移里均匀抽一条。这样每条转移被抽中的概率正比于 $1/B(s)$，每个状态获得的总采样质量相同。权重恒为 1，不需要 IS。

```text
for each stored transition i: s_i = state(obs_i)
sample s ~ Uniform({s : B(s) > 0})
sample i ~ Uniform({i : s_i = s})
update Q with weight 1
```

### 表格环境：有效

<figure>
  <img src="{{ '/assets/images/per-reeval/05_fixed_uniform_invfreq_msve.png' | relative_url }}">
  <figcaption>图 5: 固定学习率, 不同 batch 下, uniform 与 invFreq 的 MSVE 曲线</figcaption>
</figure>

<figure>
  <img src="{{ '/assets/images/per-reeval/06_tuned_uniform_invfreq_msve.png' | relative_url }}">
  <figcaption>图 6: 调优学习率, 不同 batch 下, uniform 与 invFreq 的 MSVE 曲线</figcaption>
</figure>


在表格链上，invFreq 采样最差的收敛情况是也与 Uniform 相当，显著优于 PER。

### 神经网络：优势保持

| 项目 | 设置 |
| --- | --- |
| 算法 | DQN，update freq=500, $\gamma = 0.99$，buffer size = 100k |
| 采样 | uniform / invFreq|


<figure>
  <img src="{{ '/assets/images/per-reeval/07_fixed_lr0p001_msve.png' | relative_url }}">
  <figcaption>图 7: 固定学习率, 不同 batch 下, uniform 与 invFreq 的 MSVE 曲线</figcaption>
</figure>

<figure>
  <img src="{{ '/assets/images/per-reeval/08_tuned_msve.png' | relative_url }}">
  <figcaption>图 8: 调优学习率, 不同 batch 下, uniform 与 invFreq 的 MSVE 曲线</figcaption>
</figure>

在 MarkovChain50 环境下把 Q 表替换为神经网络后 invFreq 采样仍然可以保持优势。

可见在这个简单的 Toy 环境下，基于 GPI 框架，以 MSVE 为指标，invFreq 确实更胜一筹，参考附录中图7、8对应的 replay hotmap 可以发现，invFreq 保证了回放池状态的均匀访问。

### 二维网格世界：推广尝试

MarkovChain 是一个一维状态链，现在使用二维状态链进行验证，从需要估计的状态数角度让二维与一维等价(探索难度并不相等)

| 项目 | 设置 |
| --- | --- |
| 环境 | FrozenLake-v1，边长为 7，稀疏奖励，确定性转移 |
| 指标 | MSVE：$V(s) = \max_a Q(s,a)$ 与真实 $V^*(s)$ 在非终止状态上的均方误差 |


<figure>
  <img src="{{ '/assets/images/per-reeval/09_fixed_lr0p001_msve.png' | relative_url }}">
  <figcaption>图 9: 固定学习率, 不同 batch 下, uniform 与 invFreq 的 MSVE 曲线</figcaption>
</figure>

<figure>
  <img src="{{ '/assets/images/per-reeval/010_tuned_msve.png' | relative_url }}">
  <figcaption>图 10: 调优学习率, 不同 batch 下, uniform 与 invFreq 的 MSVE 曲线</figcaption>
</figure>

二维状态链下，优势似乎被削弱了，在调优学习率下，batch 的增大会将 invFreq 的优势抹去

## 8. 总结

本次实验没有得出更有价值的结论或方法，只是对已有的问题进行了探究。为什么直觉上很好的想法实践中却不尽如人意，尚未有明确的证据或理论。

在新场景下，如果一种回放机制需要我们进行实验后才能证明其是否有优越性, 那该方法存在的意义是什么？现有经验回放机制大多如此, 我们拥有各种各种的回放机制, 却很难有准则指出什么场景下该使用何种机制。**如果不存在一种回放方法在所有场景下都更具优越性的话，或许我们最需要的是一种可靠的准则。**

## 参考文献

1. T. Schaul, J. Quan, I. Antonoglou, D. Silver. *Prioritized Experience Replay*. ICLR 2016. [arXiv:1511.05952](https://arxiv.org/abs/1511.05952)
2. A. A. Li, Z. Lu, C. Miao. *Revisiting Prioritized Experience Replay: A Value Perspective*. 2021. [arXiv:2102.03261](https://arxiv.org/abs/2102.03261)
3. M. G. Mattar, N. D. Daw. *Prioritized memory access explains planning and hippocampal replay*. Nature Neuroscience, 2018.
4. A. W. Moore, C. G. Atkeson. *Prioritized Sweeping: Reinforcement Learning with Less Data and Less Time*. Machine Learning, 1993.


## 附录

### 补充的实验结果

<figure>
  <img src="{{ '/assets/images/per-reeval/01_fixed_uniform_per_sampling_heatmap.png' | relative_url }}">
  <figcaption>图 1 对应的采样热力图 </figcaption>
</figure>

<figure>
  <img src="{{ '/assets/images/per-reeval/02_tuned_uniform_per_sampling_heatmap.png' | relative_url }}">
  <figcaption>图 2 对应的采样热力图 </figcaption>
</figure>

<figure>
  <img src="{{ '/assets/images/per-reeval/03_fixed_uniform_per_nois_sampling_heatmap.png' | relative_url }}">
  <figcaption>图 3 对应的采样热力图 </figcaption>
</figure>

<figure>
  <img src="{{ '/assets/images/per-reeval/04_tuned_uniform_per_nois_sampling_heatmap.png' | relative_url }}">
  <figcaption>图 4 对应的采样热力图 </figcaption>
</figure>

<figure>
  <img src="{{ '/assets/images/per-reeval/05_fixed_uniform_invfreq_sampling_heatmap.png' | relative_url }}">
  <figcaption>图 5 对应的采样热力图 </figcaption>
</figure>

<figure>
  <img src="{{ '/assets/images/per-reeval/06_tuned_uniform_invfreq_sampling_heatmap.png' | relative_url }}">
  <figcaption>图 6 对应的采样热力图 </figcaption>
</figure>

<figure>
  <img src="{{ '/assets/images/per-reeval/07_fixed_lr0p001_sampling_heatmap.png' | relative_url }}">
  <figcaption>图 7 对应的采样热力图 </figcaption>
</figure>

<figure>
  <img src="{{ '/assets/images/per-reeval/08_tuned_sampling_heatmap.png' | relative_url }}">
  <figcaption>图 8 对应的采样热力图 </figcaption>
</figure>

