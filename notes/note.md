能用牛顿法吗（二阶导）？不一定能求得到，就算可以，也要一个 nxn 的空间和计算，并不会很容易（一般只能做近似）

SoftMax 和 Cross Entropy Loss 的关系

- softmax 是一种将数值转化成概率的方法，同时所有的效率和为1：$\hat{y_i} = \frac{e^{o_i}}{\sum_k e^{o_k}}$
- cross entropy 是用来很亮两个概率的区别大小：$l(y, \hat{y}) = -\displaystyle \sum_i y_i \log \hat{y_i} = -\log \hat{y_y}$ 因为只有真实类别的这一项的 $y_i$ 不为0（结果就是我们不关心非真实类别的值的大小，只需要真实类别给出的值够大）
- softmax 和 cross entropy 的关系是，cross entropy loss 的梯度是 $\frac{\partial l(y, \hat{y})}{\partial o_i} = \text{softmax}(o)_i - y_i$。也就是说损失函数对于原始值求导会得到 softmax 过后的概率 - 真实值（所以能很好使用这个损失梯度来更新参数）

似然函数 Likelihood function：可以理解成损失梯度越小的地方，似然函数的值就越大

Huber's Robust Loss

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