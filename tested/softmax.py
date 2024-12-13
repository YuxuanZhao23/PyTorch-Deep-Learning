import torch
from torch import nn
from d2l import torch as d2l
from utils.utils import load_data_fashion_mnist, train_ch3, init_weights

batch_size = 256
lr = 0.1
num_epochs = 10

# data
train_iter, test_iter = load_data_fashion_mnist(batch_size)

# model
net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))
net.apply(init_weights)
loss = nn.CrossEntropyLoss(reduction='none')
trainer = torch.optim.SGD(net.parameters(), lr=lr)

# train
train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
d2l.plt.savefig('./results/softmax.png')
