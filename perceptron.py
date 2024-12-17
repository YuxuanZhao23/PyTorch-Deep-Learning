import torch
from torch import nn
from utils.utils import load_data_fashion_mnist, train_ch3, init_weights
from d2l import torch as d2l

batch_size = 256
lr = 0.1
dr = 0.1 # 0.5, 0.1, 0.9
wc = 0.001
num_epochs = 10

# data
train_iter, test_iter = load_data_fashion_mnist(batch_size)

# model
net = nn.Sequential(nn.Flatten(), nn.Linear(784, 256), nn.ReLU(), nn.Dropout(dr), nn.Linear(256, 10))
net.apply(init_weights)
loss = nn.CrossEntropyLoss(reduction='none')
trainer = torch.optim.SGD(net.parameters(), weight_decay=wc, lr=lr)

# train
train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
d2l.plt.savefig('./results/perceptron.png')