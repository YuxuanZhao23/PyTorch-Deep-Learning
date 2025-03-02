from torch.utils.data import DataLoader
import torchvision
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10(root='./CIFAR10', train=False, transform=torchvision.transforms.ToTensor(), download=True)
dataloader = DataLoader(dataset=dataset, batch_size=64, shuffle=True, drop_last=True)
writer = SummaryWriter('logs')

for i, data in enumerate(dataloader):
    images, labels = data
    writer.add_images('test dataset', images, i) # 很多张图片的时候用 add_images()

writer.flush()
writer.close()