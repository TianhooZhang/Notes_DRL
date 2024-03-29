# 多臂赌博机

**强化学习与其它机器学习方法最大的不同**:强化学习的训练信号是用评估给定动作的好坏的，而不是通过正确范例进行直接指导的。

- 评估性反馈：表明当前采取动作的好坏程度
- 指导性反馈：表明当前应采取的最好的动作

```我个人理解为评估性反馈是对过程的反馈，即基于某一个原则对每一个动作进行评价，而指导性反馈只是告诉你，你应该去做什么，而不说你当前做的与你应该做的之间的关系。举个例子，假设一道题是10分，你有其中一步做错了，评估性老师会告诉你当前得了7分，而指导性老师会告诉你，你得了0分。```

所以说，评估性反馈依赖于当前采取的动作，即采取不同的动作会得到不同的反馈；而指导性反馈则不依赖于当前采取的动作，即采取不同的动作也会得到相同的反馈。

本章节主要是通过讨论”K臂赌博机问题“，一个典型的非关联的评估性反馈问题来为之后的关联的完全强化学习问题做基础。

```非关联的：动作不会使环境发生改变。```

## 2.1 一个k臂赌博机问题

问题描述：重复在k个选项或动作中进行选择，每次选择会有一个收益，每个动作的收益服从一种分布。目标是在某一段时间内最大化总收益的期望。
我们称某种动作带来的收益的期望为这个动作的*价值*。
记 $t$ 时刻选择的动作为 $A_t$ ，收益为 $R_t$ ，任意动作 $a$ 的价值为 $q_*(a)$ ，即 $q_*(a) \dot= \mathbb{E}[R_t|A_t=a]$ 。
记动作 $a$ 在时刻 $t$ 的价值的估计为 $Q_t(a)$，我们期望 $Q_t(a)$ 接近 $q_*(a)$。

如果持续对动作的价值进行估计，那么在任一时刻都会至少有一个动作的估计价值是最高的，我们将这些对应最高估计价值的动作成为**贪心动作**。如果选择贪心动作，即称之为**开发**，如果不是基于贪心选择动作，而是随机选择动作，即为**试探**。
开发会最大化当前动作的收益，而试探则会带来总体收益的提升。
```这个很好理解，一个是守旧一个是创新，守旧会带来当前动作更为准确的价值估计，而创新则会提高其它动作的价值估计，从而准确得知当前动作相对于其它动作的价值估计。```
强化学习需要去解决的一个问题就是：**开发与试探的平衡**。
本文先不讨论如何去平衡开发与试探，而是通过实验验证：平衡优于贪心，也即*开发+试探>开发*。

## 2.2 动作-价值方法

使用价值估计来进行动作选择的方法称之为”动作-价值方法“。
由于动作价值的真实值是选择这个动作时的期望收益，因此最简单的就是通过计算实际收益的平均值来估计动作的价值。
$Q_t(a) \dot= \frac{t时刻前通过执行动作a得到的收益总和}{t时刻前执行动作a的次数}=\frac{\sum_{i=1}^{t-1}R_i·\mathbb{I}_{A_i=a}}{\sum_{i=1}^{t-1}\mathbb{I}_{A_i=a}}$
$\mathbb{I_{predicate}}$表示随机变量，当$predicate$为真时，其值为1，反之为0。
当分母为$0$时，设置$Q_t(a)$为某个常数，比如说$Q_t(a)=0$。
根据大数定律，$Q_t(a)$会收敛到$q_*(a)$。

可以根据**贪心原则**选择动作，即选择具有最高估计值的动作。
$A_t \dot= argmax_aQ_t(a)$
其中$argmax_a$ 是使得$Q_t(a)$的值最大的动作$a$。

还可以在贪心原则上加入探索，形成 **$\epsilon$-贪心** 策略。
即以$\epsilon$概率从所有动作中随机选择一个动作，以$1-\epsilon$的概率按照贪心来选择动作。
由此我们可以看出，在无限次选择动作时，每个动作都会被无限次的采样，从而满足所有动作的估计价值逼近其实际价值。最终你那个最优动作的选择概率会大于$1-\epsilon$。

## 2.3 10臂测试平台

代码详见code_2_3.py

## 2.4 增量式实现

在之前的内容中，会把动作收益的样本均值作为该动作的动作价值的估计，但是正如*code_2_3.py*中实现那样，这种估计方法需要常数级的内存来保存一些数据，比如说该动作总的价值以及采取该动作的次数。
我们讨论一种新的方式来解决该问题，即将平均计算改为增量式计算。
已知：
$Q_n \dot= \frac{R_1+R_2+\ldots+R_{n-1}}{n-1}$
得：
$$
Q_{n+1} = \begin{cases}
\frac{1}{n}\sum_{i=1}^{n}R_i \\
\frac{1}{n}(R_n+\sum_{i=1}^{n-1}R_i) \\
\frac{1}{n}(R_n + (n-1)\frac{1}{n-1}\sum_{i=1}^{n-1}R_i) \\
\frac{1}{n}(R_n + (n-1)Q_n) \\
\frac{1}{n}(R_n+nQ_n-Q_n) \\
Q_n + \frac{1}{n}[R_n - Q_n]
\end{cases}
$$
在这种情况下，只需要存储$Q_n$和$n$。从上式可以推理出：
**新估计值  <-- 旧估计值 + 步长 × [目标 - 旧估计值]**
其中*[目标 - 旧估计值]*是估计值的误差，误差会随着目标靠近而减少。
在上式中，处理动作a的第n个收益的步长是1/n，而我们更通用$\alpha$。
因此，我们可以对*code_2_3.py*进行修改，得到*code_2_4.py*
两者的区别仅仅是更换了更新方法。

## 2.5 跟踪一个非平稳问题

我们之前讨论的取平均方法对平稳的赌博机问题是合适的，即收益的概率从不随着时间变化的赌博机问题。但如果赌博机的收益率是随着时间变化的，该方法是不合适的，这种收益率随时间变化的问题我们称之为**非平稳问题**，这也是强化学习经常遇到的。

对于非平稳问题，给近期的收益赋予比过去很久的收益更高的权重是一种合理的处理方式，最流行的方法是**固定步长**。故而更新公式从
$$
Q(A) \leftarrow Q(A) + \frac{1}{N(A)}[R - Q(A)]
$$
更换为：
$$
Q_{n+1} \dot= Q_n + \alpha[R - Q_n]
$$
我们对公式进行拆解，将$Q_{n+1}$拆为仅和$n.Q_1,\alpha,R_i$有关的式子，得：
$$
Q_{n+1} = (1-\alpha)^nQ_1 + \sum_{i=1}^n \alpha(1-\alpha)^{n-i}R_i
$$
由于$0<\alpha<1$，所以距离当前时刻$n$越远，其$R$的重要程度越小。
又因为权重之和$(1-\alpha)^n + \sum_{i=1}^n \alpha(1-\alpha)^{n-i} = 1$，所以我们称为**加权平均**，更准确是**指数近因加权平均**。

根据**随机逼近**理论，保证收敛概率为1所需的条件为：
$$
\sum_{n=1}^\infty\alpha_n(a) = \infty,且 \sum_{n=1}^\infty\alpha_n^2 < \infty
$$
对于采样平均中$\alpha_n(a)=\frac{1}{n}$来说，满足上述两个条件，因为$\frac{1}{n}$的累加和趋向于无穷，而$\frac{1}{n^2}$的累加和趋向于一个常数。但是对于常数步长$\alpha_n(a)=\alpha$中，则不满足第二个条件，其平方的和也趋向于无穷，说明估计无法完全收敛，而是会随着最近得到的收益而变化。

## 2.6 乐观初始值

在开本节前，要先说一下*无偏估计*与*有偏估计*，何为无偏估计，即估计量的数学期望等于被估计量的真实值，有偏估计则意味着不等于被估计量的真实值。

对于之前提到的算法中，整个更新公式都与估计值的初始值有关，即依赖于$Q_1$，这个$Q_1$是一个人为经验，我们也可以理解为超参数，所以估计最初是有偏的，但是对于$1/n$的学习率来说，所有的动作都会被选择，在无穷时，每个动作的估计期望都会收敛于真实值，所以是无偏估计的，但是对于学习率是定常来说，其无法收敛，所以偏差会随时间减小但不会消失。

不难看出，初始动作的估计价值提供了一种简单的试探方法，比如说在之前的任务中，我们将初始的价值由0改为5，由于奖励服从均值为0方差为1的正态分布，所以初始值要远高于任何一个动作的期望价值，所以每个动作在初期都会被试探而不是开发，我们称这种鼓励试探的技术为**乐观初始价值**。

## 2.7 基于置信度上界的动作选择

在本文中给出了一个新的选择动作的方式，即基于置信度上界（UCB）的动作选择。
$$
A_t \dot= \argmax_a[Q_t(a) + c\sqrt{\frac{\ln{t}}{N_t(a)}}]
$$
我们可以基于函数的性质来理解这个公式，$N_t(a)$意味着选择动作$a$的次数，其在分母上，所以说次数越少，选择的概率越大，$\ln{t}$意味着随着时间的增加，选择概率会逐渐增加。所以说，当$\ln{t}$一定时，会优先选择之前没怎么选择过的动作，而当$N_t(a)$一定时，随着t的增加，该动作的选择概率也会增加。
```code_2_7.py加入了UCB公式```

## 2.8 梯度赌博机算法

前期我们直接使用评估值来选择动作。在本节，我们引入偏好函数$H_t(a)$来替代之前讲述的直接通过贪婪来选择动作，偏好函数越大，动作被选择的概率越大。将直接选择最大估计价值的动作更替为：
$$
Pr\{A_t = a\} \dot= \frac{e^{H_t(a)}}{\sum_{b=1}^k e^{H_t(b)}} \dot= \pi_t(a)
$$
这是一个典型的softmax函数，每个动作的偏好同时加上同一个数并不会对概率有影响，所以我们在这里考虑的是动作之间的相对偏好。

**其中$\pi_t(a)$是在t时刻选择a的概率**。

对于动作a的更新：
$$
H_{t+1}(A_t) \dot= H_t(A_t) + \alpha(R_t - \overline{R}_t)(1-\pi_t(A_t))
$$
对于动作不是a的更新：
$$
H_{t+1}(A_t) \dot= H_t(A_t) - \alpha(R_t - \overline{R}_t)\pi_t(a), a \not ={A_t}
$$

证明的核心为：

- 所有梯度的和为0
- $\mathbb{E}(q_*(A_t)) = \mathbb{E}(R_t)$

```代码见code_2_8.py```

## 2.9 关联搜索（上下文相关的赌博机）

这节主要是为了向完全强化学习过度，即此时的动作会影响下次的情境和收益。

## 习题

### 练习2.1

$P_{贪心动作被选择}=0.5+0.5\times0.5=0.75$
```第一个0.5是贪心选择的概率，第二个0.5是试错时选择```

### 练习2.2 赌博机的例子

肯定发生的情境：第四步选择2号臂肯定是随机的，第五步选择3号臂也肯定是随机的，其余情况都是有可能是贪心的也有可能是随机的。

```分析：
- Step0：$Q_1(a) = 0, \forall{a}$
- Step1: $A_1 = 1, R_1 = -1$, $Q_2(1) = -1$
- Step2: $A_2 = 2, R_2 = 1$, $Q_3(2) = 1$
- Step3: $A_3 = 2, R_3 = -2$, $Q_4(2) = -0.5$
- Step4: $A_4 = 2, R_4 = 2$, $Q_5(2) = 1/3$
- Step5: $A_5 = 3,, R_5 = 0$, $Q_6(3) = 0$
```

``` 解答思路：
- 第一步选择了1号机器，有可能是贪心，也有可能是随机，因为所有动作的估计价值都是0，无论是贪心选还是随机选，都会是从这几个动作中随机选出一个，所以第一个动作不能肯定。
- 第二步选择了2号机器，有可能是贪心，也有可能是随机，因为第一步之后，1号机器的q估计值已经成为了-1，而其它机器的q估计值都为0，所以选择2号机器两种都有可能。
- 第三步选择2号机器，有可能是贪心，也有可能是随机，因为第二步之后，2号机器的q估计值变为了1，成为了几个机器最大的，如果是贪心即选择2号机器，如果不是贪心，也有可能选择到第2号机器。
- 第四步选择2号机器人，一定是随机的，因为第三步选择2号机器后，2号机器的价值估计为-0.5，而3号和4号的机器都是0，所以按照贪心不可能选择到2号机器，只能是随机选择到的。
- 第五步是选择3号机器，一定是随机的，因为第四步选择完2号机器后，2号机器的价值估计为0.33，而3号和4号的机器都是0，所以按照贪心策略应该选择2号机器，而非3号机器，所以选择到3号机器一定是贪心的。
```

### 练习2.3

$\epsilon = 0.01$会表现的更好，其选择最优动作的概率为0.991，而$\epsilon = 0.1$选择最优动作的概率是0.91，所以前者的效果超出了后者
$(0.991-0.91)/0.91 \dot= 0.089$。

```分析
该题问的是长期来看，也就是说，当t趋向于无穷时哪种效果会表现好一些。
显而易见是e-贪心算法要由于贪心算法，那么对于e的选择来说，当算法收敛时，e=0.1时，对于10臂老虎机，有1-0.1+0.1*0.1=0.91的概率选择最优动作，
e=0.01时，对于10臂老虎机，有1-0.01+0.01*0.1=0.991的概率选择最优动作，
所以其动作是其1.089倍
```

### 练习2.4

我们令$\alpha_0 = 1, 得到如下公式$
$$
Q_{n+1} = \prod_{i=1}^n(1-\alpha_i)Q_1 + \sum_{i=1}^n \alpha_i \cdot R_i \cdot \prod_{k=i+1}^n(1-\alpha_k)
$$
且当$x>y$时，$\prod_{i=x}^yf(i) \dot= 1$。

```这道题的本质还是把定常数的步长改为不定常数的步长来对书中2.6公式进行推导，稍微推导一下就可以写出。```

### 练习2.5

详见代码code_2_5.py。

### 练习2.6

因为乐观的初始值导致了更多的探索，在初期算法会通过前期的随机选择，尽快找到最优解，但是由于初始值设置为5，更细步长较小，所以算法需要一定的时间将价值估计逐渐从5收敛到-1~1，所以前期震荡较，但是从全局上来看，更多考虑了所有动作的价值，会更加准确。

### 练习2.7

首先我们将
$$
\beta_n \dot= \alpha/ \overline{o}_n
$$
代入到练习2.4我们得到的通用公式中，得到了：
$$
Q_{n+1} = \prod_{i=1}^n(1-\beta_i)Q_1 + \sum_{i=1}^n \beta_i \cdot R_i \cdot \prod_{k=i+1}^n(1-\beta_k)
$$
我们来看，当n=1时，$Q_2 = (1-\beta_1)Q_1 + \beta_1 \cdot R_1$
而，$\beta_1 = \alpha/\overline{o}_1$,其中$\overline{o}_n \dot= \overline{o}_{n-1} + \alpha(1-\overline{o}_{n-1})$,且$\overline{o}_0 =0$，
即，$\beta_1 = \alpha/\overline{o}_1 = \alpha/\alpha = 1$,
所以说，$1-\beta_1 = 0$，
$$
Q_{n+1} = \sum_{i=1}^n \beta_i \cdot R_i \cdot \prod_{k=i+1}^n(1-\beta_k)
$$
即**对初始值是无偏**的。
然后我们再证明一下，近期动作奖励的权重要大于远期动作奖励的权重。
令：
$$
\omega_i = \beta_i \cdot \prod_{k=i+1}^n(1-\beta_k)
$$
，则:
$$
\frac{\omega_{i+1}}{\omega_i} = \frac{\beta_{i+1}}{\beta_i(1-\beta_{i+1})} = \frac{1}{\frac{\beta_i}{\beta_{i+1}
}(1-\beta_{i+1})}
$$
单独看分母：
$$
\frac{\beta_i}{\beta_{i+1}}(1-\beta_{i+1}) = \frac{\overline{o}_{i+1}}{\overline{o}_i}(\frac{\overline{o}_{i+1}-\alpha}{\overline{o}_{i+1}}) = \frac{\overline{o}_{i+1}-\alpha}{\overline{o}_i} = \frac{\overline{o}_i + \alpha(1-\overline{o}_i)-\alpha}{\overline{o}_i} = 1 - \alpha
$$
所以说，
$$
\frac{\omega_{i+1}}{\omega_i} = \frac{1}{1-\alpha} > 1
$$
所以近期动作奖励的权重要大于远期动作奖励的权重，认为其是对初始值无偏的指数近因加权平均。

### 练习2.8

因为在最初的10步，每一个动作都会被考虑一次，具体是因为UCB公式会优先选择从来没有选择过的动作。在第10步之后，所有动作都被选择过一次了，就会进入UCB贪心选择。当一个动作的贪心值小于其它动作随着$t$增加导致的$\sqrt{\frac{\ln{t}}{N_t(a)}}$时，就会进入随机选择。所以reward会减少。
但是当$t \rightarrow \infty$时，探索会趋向于0，之后会一直贪婪。
c的增加提升了前期探索的可能性。

### 练习2.9

根据题意，因为是logistic回归和sigmoid回归，所以其输出为0，1。
我们设定a=0,1。则
$$
Pr\{A_t=1\} \dot= \frac{e^{H_t(1)}}{e^{H_t(0)}+e^{H_t(1)}} = \frac{1}{1 + e^{-x}}, x= H_t(1) - H_t(0)
$$

### 练习2.10

第一问是0.5，因为动作选择概率相同，都是0.5，而每个动作的期望也是0.5，所以最优期望也是0.5
第二问是0.55，因为在给定场景下，就知A选0.2，B选0.9，所以是0.55。

### 练习2.11

可以根据前面编写的code文件自行比较。