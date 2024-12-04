import random
import torch

def create_synthetic_data(w, b, num_examples):
    X = torch.normal(0, 1, (num_examples, len(w))) # 定义均值为0，方差为1，大小为样本数 x 每个样本的（这里可以是随机选）
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape) # 加入噪音均值为0，方差为0.01（噪音一般用正态分布）
    return X, y.reshape((-1, 2)) # 把 y 变成列向量

true_w = torch.tensor([[2, -3.4], [5, 1], [7, 9]])
true_b = torch.tensor([4.2, 3])
features, labels = create_synthetic_data(true_w, true_b, 1000)

def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices) # 随机打乱的序号

    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(indices[i:min(i + batch_size, num_examples)]) # 这里用 min 来处理如果最后一个 batch 长度不足的 edge case

        # tensor([features[5], features[3], features[0], features[11]])
        yield features[batch_indices], labels[batch_indices] # 每次提供一个 batch 的随机序号的 feature 和对应的 label

def linreg(X):
    return torch.matmul(X, w) + b

def mean_squared_loss(y_hat, y):
    return ((y_hat - y) ** 2 / 2).mean()

def sgd(params, lr):
    with torch.no_grad(): # 暂时关闭 autograd 检查，从而使用 in-place 更改
        for param in params:
            param -= (lr * param.grad) # 这样写才是 in-place，如果直接用 param = param - (lr * param.grad) 会创建新的 param 从而丢失 grad
            param.grad.zero_()

batch_size = 10

w = torch.normal(0, 0.01, size=(3, 2), requires_grad=True) # 初始化成噪音均值为0，方差为0.01，定义可导，所以backward的时候更新的是这个
b = torch.zeros(2, requires_grad=True) # b 其实是一个标量

lr = 0.03
num_epochs = 5
net = linreg
loss = mean_squared_loss

for epoch in range(num_epochs): # 要在数据上跑多少次
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X), y) # 正着算 loss
        l.backward() # l是长度为1的tensor，会更新w
        sgd([w, b], lr)
    train_l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')