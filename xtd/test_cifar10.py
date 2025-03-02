import torchvision
from torch.utils.tensorboard import SummaryWriter

dataset_transform = torchvision.transforms.ToTensor()

train_dataset = torchvision.datasets.CIFAR10('./CIFAR10', train=True, transform=dataset_transform, download=True)
test_dataset = torchvision.datasets.CIFAR10('./CIFAR10', train=False, transform=dataset_transform, download=True)

sw = SummaryWriter('logs')
for i in range(10):
    sw.add_image(train_dataset.classes[train_dataset[i][1]], train_dataset[i][0])