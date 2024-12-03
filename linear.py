import torch
from re import L
from torch import nn
from torch.utils import data
import random

from_scratch = True

# data
def create_synthetic_data(w, b, num_examples):
  X = torch.normal(0, 1, (num_examples, len(w))) # 定义均值为0，方差为1，大小为样本数 x 每个样本的（这里可以是随机选）
  y = torch.matmul(X, w) + b
  y += torch.normal(0, 0.01, y.shape) # 加入噪音均值为0，方差为0.01（噪音一般用正态分布）
  return X, y.reshape((-1, 1)) # 把 y 变成列向量

true_w = torch.tensor([2, -3.4])
true_b = torch.tensor(4.2)
features, labels = create_synthetic_data(true_w, true_b, 1000)

def load_array_from_scratch(features, labels, batch_size):
  num_examples = len(features)
  indices = list(range(num_examples))
  random.shuffle(indices) # 随机打乱的序号

  for i in range(0, num_examples, batch_size):
    batch_indices = torch.tensor(indices[i:min(i + batch_size, num_examples)]) # 这里用 min 来处理如果最后一个 batch 长度不足的 edge case

    # tensor([features[5], features[3], features[0], features[11]])
    yield features[batch_indices], labels[batch_indices] # 每次提供一个 batch 的随机序号的 feature 和对应的 label


def load_array(features, labels, batch_size, is_train=True):
  dataset = data.TensorDataset(features, labels)
  return data.DataLoader(dataset, batch_size, shuffle=is_train)

# batch size 其实比较小是一件好事，因为越小噪音越多，反而避免一开始越走越歪
batch_size = 100
if from_scratch: data_iter = load_array_from_scratch(features, labels, batch_size)
else: data_iter = load_array(features, labels, batch_size)

# model
num_epochs = 10
lr = 0.03

if from_scratch:
  w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True) # 初始化成噪音均值为0，方差为0.01，定义可导，所以backward的时候更新的是这个
  b = torch.zeros(1, requires_grad=True) # b 其实是一个标量
  net = lambda X: torch.matmul(X, w) + b
  loss = lambda y_hat, y: ((y_hat - y) ** 2 / 2).mean()

  def sgd(params, lr):
    with torch.no_grad(): # 暂时关闭 autograd 检查，从而使用 in-place 更改
      for param in params:
        param -= (lr * param.grad) # 这样写才是 in-place，如果直接用 param = param - (lr * param.grad) 会创建新的 param 从而丢失 grad
        param.grad.zero_()

else:
  net = nn.Linear(2, 1)
  net.weight.data.normal_(0, 0.01) # 初始化 w 和 b
  net.bias.data.fill_(0)
  loss = nn.MSELoss()
  trainer = torch.optim.SGD(net.parameters(), lr=lr)

# train
for epoch in range(num_epochs):
  for X, y in data_iter:
    l = loss(net(X), y) # 正着算 loss

    if not from_scratch: trainer.zero_grad()

    l.backward() # l是长度为1的tensor，会更新w

    if from_scratch: sgd([w, b], lr)
    else: trainer.step()

  l = loss(net(features), labels)
  print(f'epoch {epoch + 1}, loss {l:f}')