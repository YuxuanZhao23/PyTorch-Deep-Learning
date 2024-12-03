import torch
import torchvision
from torch import nn
from torch.utils import data
from d2l import torch as d2l
from torchvision import transforms
from IPython import display
from utils import train_ch3

# data
def load_data_fashion_mnist(batch_size, resize=None):
  trans = [transforms.ToTensor()]
  if resize:
    trans.insert(0, transforms.Resize(resize))
  trans = transforms.Compose(trans)
  # 这里的 transform=trans 是指把图片变成 tensor
  mnist_train = torchvision.datasets.FashionMNIST(root="../data", train=True, transform=trans, download=True)
  mnist_test = torchvision.datasets.FashionMNIST(root="../data", train=False, transform=trans, download=True)
  return (data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=4),
          data.DataLoader(mnist_test, batch_size, shuffle=False, num_workers=4))

batch_size = 256
train_iter, test_iter = load_data_fashion_mnist(batch_size)

# model
num_inputs = 784
num_outputs = 10

net = nn.Sequential(nn.Flatten(), nn.Linear(num_inputs, num_outputs))

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights)

loss = nn.CrossEntropyLoss(reduction='none')
trainer = torch.optim.SGD(net.parameters(), lr=0.1)

# train
num_epochs = 10
train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
d2l.plt.savefig('softmax.png')
