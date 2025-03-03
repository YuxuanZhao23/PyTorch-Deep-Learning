from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from torch.nn import Module, Conv2d, MaxPool2d, ReLU, Sequential, Flatten, Linear, Dropout, CrossEntropyLoss
from torch.nn.init import xavier_uniform_
from torch.utils.tensorboard import SummaryWriter
from torch.optim import SGD
from torch import no_grad

class Block(Module):
    def __init__(self, num_convs, in_channels, out_channels):
        super().__init__()
        network = []
        network.append(Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1))
        network.append(ReLU())
        for _ in range(num_convs-1):
            network.append(Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1))
            network.append(ReLU())
        network.append(MaxPool2d(kernel_size=2, stride=2))
        self.network = Sequential(*network)

    def forward(self, x):
        return self.network(x)
    
class VGG(Module):
    def __init__(self):
        super().__init__()
        self.network = Sequential(Block(1, 3, 64), Block(2, 64, 128), Flatten(),
                                   Linear(128 * 8 * 8, 4096), ReLU(), Dropout(0.5),
                                   Linear(4096, 10))
        self.network.apply(self.init_weights)
        self.network.to(device)

    def forward(self, x):
        return self.network(x)
    
    def init_weights(self, model):
        if type(model) == Linear or type(model) == Conv2d:
            xavier_uniform_(model.weight)
    
def accuracy(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1: y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

train_data_set = CIFAR10('./CIFAR10', train=True, transform=ToTensor())
test_data_set = CIFAR10('./CIFAR10', train=False, transform=ToTensor())
train_data_loader = DataLoader(dataset=train_data_set, batch_size=64)
test_data_loader = DataLoader(dataset=test_data_set, batch_size=64)
device = 'cuda'
writer = SummaryWriter('logs')
vgg = VGG()
optimizer = SGD(vgg.parameters(), lr=5e-2)
loss = CrossEntropyLoss()
num_of_batch = len(train_data_loader)

for epoch in range(25):
    vgg.train()
    total_loss, total_accuracy, total = 0, 0, 0
    for i, data in enumerate(train_data_loader):
        optimizer.zero_grad()
        X = data[0].to(device)
        Y = data[1].to(device)
        y_hat = vgg(X)
        l = loss(y_hat, Y)
        l.backward()
        optimizer.step()

        with no_grad():
            total_loss += l * Y.numel()
            total_accuracy += accuracy(y_hat, Y)
            total += Y.numel()
            writer.add_scalar("VGG training loss", total_loss/total, epoch * num_of_batch + i)
            writer.add_scalar("VGG training accuracy", total_accuracy/total, epoch * num_of_batch + i)

    vgg.eval()
    with no_grad():
        total_accuracy, total = 0, 0
        for i, data in enumerate(test_data_loader):
            X = data[0].to(device)
            Y = data[1].to(device)
            total_accuracy += accuracy(vgg(X), Y)
            total += Y.numel()
    writer.add_scalar("VGG testing accuracy", total_accuracy/total, epoch)

writer.flush()
writer.close()