import torch
from re import L
from torch import nn
from torch.utils import data

# data
def create_synthetic_data(w, b, num_examples):
  X = torch.normal(0, 1, (num_examples, len(w))) # 定义均值为0，方差为1，大小为样本数 x 每个样本的（这里可以是随机选）
  y = torch.matmul(X, w) + b
  y += torch.normal(0, 0.01, y.shape) # 加入噪音均值为0，方差为0.01（噪音一般用正态分布）
  return X, y.reshape((-1, 1)) # 把 y 变成列向量

true_w = torch.tensor([2, -3.4])
true_b = torch.tensor(4.2)
features, labels = create_synthetic_data(true_w, true_b, 1000)

def load_array(features, labels, batch_size, is_train=True):
  dataset = data.TensorDataset(features, labels)
  return data.DataLoader(dataset, batch_size, shuffle=is_train)

# batch size 其实比较小是一件好事，因为越小噪音越多，反而避免一开始越走越歪
batch_size = 100
data_iter = load_array(features, labels, batch_size)

# model
num_epochs = 10
lr = 0.03
net = nn.Linear(2, 1)
net.weight.data.normal_(0, 0.01) # 初始化 w 和 b
net.bias.data.fill_(0)
loss = nn.MSELoss()
trainer = torch.optim.SGD(net.parameters(), lr=lr)

# train
for epoch in range(num_epochs):
  for X, y in data_iter:
    l = loss(net(X), y) # 正着算 loss
    trainer.zero_grad()
    l.backward() # l是长度为1的tensor，会更新w
    trainer.step()

  l = loss(net(features), labels)
  print(f'epoch {epoch + 1}, loss {l:f}')