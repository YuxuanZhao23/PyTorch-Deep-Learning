import torch
from torch import nn

class Inception(nn.Module):
  def __init__(self,in_channels,c1,c2,c3,c4):
    super(Inception,self).__init__()
    self.conv1 = nn.Conv2d(in_channels,c1,kernel_size = 1)
    self.conv2 = nn.Sequential(
                nn.Conv2d(in_channels,c2[0],kernel_size = 1),nn.ReLU(),
                nn.Conv2d(c2[0],c2[1],kernel_size=3,padding=1),nn.ReLU()
            )
    self.conv3 = nn.Sequential(
                nn.Conv2d(in_channels,c3[0],kernel_size=1),nn.ReLU(),
                nn.Conv2d(c3[0],c3[1],kernel_size=5, padding=2),nn.ReLU()
            )
    self.conv4 = nn.Sequential(
                nn.MaxPool2d(kernel_size=3,stride=1,padding=1),
                nn.Conv2d(in_channels,c4,kernel_size = 1),nn.ReLU(),
            )
  def forward(self,X):
    return torch.cat(
            (
            self.conv1(X),
            self.conv2(X),
            self.conv3(X),
            self.conv4(X),
            ),
            dim = 1
    )
  
class GoogleNet(nn.Module):
  def __init__(self,in_channels,classes):
    super(GoogleNet,self).__init__()
    self.model = nn.Sequential(
              nn.Conv2d(in_channels,out_channels=64,kernel_size=7,stride=2,padding=3),nn.ReLU(),
              nn.MaxPool2d(kernel_size=3,stride=2),
              nn.Conv2d(in_channels=64,out_channels=64,kernel_size=1),nn.ReLU(),
              nn.Conv2d(in_channels=64,out_channels=192,kernel_size=3,padding=1),nn.ReLU(),
              nn.MaxPool2d(kernel_size=3,stride=2),
              Inception(192,c1=64,c2=[96,128],c3=[16,32],c4=32),
              Inception(256,c1=128,c2=[128,192],c3=[32,96],c4=64),
              nn.MaxPool2d(kernel_size=3,stride=2,padding=1),
              Inception(480,c1=192,c2=[96,208],c3=[16,48],c4=64),
              Inception(512,c1=160,c2=[112,224],c3=[24,64],c4=64),
              Inception(512,c1=128,c2=[128,256],c3=[24,64],c4=64),
              Inception(512,c1=112,c2=[144,288],c3=[32,64],c4=64),
              Inception(528,c1=256,c2=[160,320],c3=[32,128],c4=128),
              nn.MaxPool2d(kernel_size=3,stride=2,padding=1),
              Inception(832,c1=256,c2=[160,320],c3=[32,128],c4=128),
              Inception(832,c1=384,c2=[192,384],c3=[48,128],c4=128),
              nn.AvgPool2d(kernel_size=7,stride=1),
              nn.Dropout(p=0.4),
              nn.Flatten(),
              nn.Linear(1024,classes),
              nn.Softmax(dim=1)
            )
  def forward(self,X:torch.tensor):
    for layer in self.model:
      X = layer(X)
      print(layer.__class__.__name__,'output shape:',X.shape)

X = torch.randn(size=(1,3,224,224))
net = GoogleNet(3,1000)
net(X)