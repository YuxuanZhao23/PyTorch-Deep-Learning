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