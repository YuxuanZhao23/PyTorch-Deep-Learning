# 牛顿法

能用牛顿法吗（二阶导）？不一定能求得到，就算可以，也要一个 nxn 的空间和计算，并不会很容易（一般只能做近似）

# SoftMax 和 Cross Entropy Loss 的关系

- softmax 是一种将数值转化成概率的方法，同时所有的效率和为1：$\hat{y_i} = \frac{e^{o_i}}{\sum_k e^{o_k}}$
- cross entropy 是用来很亮两个概率的区别大小：$l(y, \hat{y}) = -\displaystyle \sum_i y_i \log \hat{y_i} = -\log \hat{y_y}$ 因为只有真实类别的这一项的 $y_i$ 不为0（结果就是我们不关心非真实类别的值的大小，只需要真实类别给出的值够大）
- softmax 和 cross entropy 的关系是，cross entropy loss 的梯度是 $\frac{\partial l(y, \hat{y})}{\partial o_i} = \text{softmax}(o)_i - y_i$。也就是说损失函数对于原始值求导会得到 softmax 过后的概率 - 真实值（所以能很好使用这个损失梯度来更新参数）

# 似然函数 Likelihood function

可以理解成损失梯度越小的地方，似然函数的值就越大

# Huber's Robust Loss

$$
l(y, y')
\left\{
\begin{aligned}
|y - y'| - \frac{1}{2}, &|y - y'| > 1 \\
\frac{(y - y')^2}{2}, &\text{ otherwise}
\end{aligned}
\right.
$$

优点在于处处可导，而且接近最优解/likelihood最大的地方的时候导数绝对值缓慢平滑变小

# 权重衰退

$$\frac{\partial (l(w, b) + \frac{\lambda ||w||^2}{2})}{\partial w} = \frac{\partial l(w, b)}{\partial w} + \lambda w$$

$$w_{t + 1} = (1 - \eta \lambda) w_t - \frac{\eta \partial l(w_t, b_t)}{\partial w_t}$$

这里的 $\eta \lambda \in (0, 1)$，也就是每次学习的时候都先对权重进行缩小，所以我们叫权重衰退

可以理解成使用 L2 norm 来限制模型的容量，可能会得到泛化性能更好的模型（不允许模型去拟合 outlier）

这个的效果很有限

# dropout

无偏差地加入噪音

$$x_i' = \left\{
\begin{array}{c c}
0 & p \\
\frac{x_i}{1 - p} & 1-p
\end{array}
\right.$$

原本以为是ensemble多个小模型，但实际上起到的是正则项 regularization 的效果

主要用在多层感知机的隐藏层输出上，不在CNN中使用

dropout 会让收敛变慢，但是学习率也不用对应调高（多训练一会儿就好）

# 数值稳定性

很深的神经网络在做 back propagation 的时候会做很多次矩阵乘法，所以会有梯度爆炸和梯度消失的问题

梯度爆炸的问题：
- infinity
  - 尤其是在16位的浮点数 6e-5 ~ 6e4
- 对学习率敏感
  - 学习率大：参数大：梯度大
  - 学习率小：训练不了
  - 可能需要训练过程动态调整学习率

体现在准确率可能会稳定在50%，基本上什么都没学会

使用 ReLU 和 Xavier initialization weight, resnet

# 卷积

平移不变性：所谓的2维卷积，也就是交叉相关

局部性：太远的参数设为0

一般来说 kernel 大小是最重要的，padding 一般用默认的，stride 在需要减少运算量的时候使用

为什么现在pooling用得越来越少？
- 因为只是想要减少运算的话，用stride conv就好了。
- 我们会做 data augmentation，所以不太需要pooling的平移性了

pooling可以理解成一种较弱的regularization（比起drop out或者L1/L2 regularization）因为可以减小dimension和feature variance，所以达到了限制模型容量的效果

# Batch Norm

对于很深的网络来说，loss 出现在最后，后面的层会学习得比较快。靠近数据的前面的层训练得很慢，但是前面的层变了之后，后面的层又需要相应的改变。最终收敛的速度变慢。

我们可以在学习前面的层的时候，锁定后面的层吗？

批量归一化能加速收敛，但是不会改善精度。可以理解成加了随机scale和noise，所以一般不会和dropout同时使用

$$BN(x) = \gamma \circ \frac{x - \mu}{\sigma} + \beta$$

将权重减去平均值，再除以方差，乘上线性系数并加上偏移（这两个是可以学习的参数）

全连接中计算的是特征维的，卷积层中计算的是通道维的。（意思是说除了这一维以外，对其他所有维度求均值）

代码部分放在LeNet了（因为对channel求均值，所以norm大小是另外的所有维）