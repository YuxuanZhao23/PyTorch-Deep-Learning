from torchvision.models import vgg16
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor, Compose, Resize
from torch.nn import Linear, CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch import no_grad
from torch.utils.tensorboard import SummaryWriter

def accuracy(outputs, targets):
    if outputs.dim() > 1 and outputs.size(1) > 1:
        outputs = outputs.argmax(dim=1)
    return (outputs == targets).sum().item()

batch_size = 64
num_epochs = 5
device = 'cuda'

writer = SummaryWriter('logs')

transform = Compose([Resize((224, 224)), ToTensor()])

train_dataset = CIFAR10('./CIFAR10', train=True, transform=transform, download=True)
test_dataset = CIFAR10('./CIFAR10', train=False, transform=transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

VGG16_CIFAR10 = vgg16(weights='DEFAULT')

for param in VGG16_CIFAR10.parameters():
    param.requires_grad = False

# 1. 在原本 1000 类的分类头之后再放一个 10 类的分类头：效果不好
# VGG16_CIFAR10.classifier.add_module('add_linear', Linear(1000, 10))

# 2. 替换原本 1000 类的分类头为一个 10 类的分类头：从模型的结构来说更合理
VGG16_CIFAR10.classifier[6] = Linear(4096, 10)
for param in VGG16_CIFAR10.classifier[6].parameters():
    param.requires_grad = True

loss_function = CrossEntropyLoss()
optimizer = Adam(VGG16_CIFAR10.classifier[-1].parameters(), lr=1e-3)

VGG16_CIFAR10.to(device)

for epoch in range(num_epochs):
    VGG16_CIFAR10.train()
    running_loss, running_correct, running_samples = 0, 0, 0
    for i, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = VGG16_CIFAR10(inputs)
        batch_loss = loss_function(outputs, targets)
        batch_loss.backward()
        optimizer.step()
        
        running_loss += batch_loss * targets.size(0)
        running_correct += accuracy(outputs, targets)
        running_samples += targets.size(0)

    train_loss = running_loss / running_samples
    train_acc = running_correct / running_samples

    VGG16_CIFAR10.eval()
    test_correct, test_samples = 0, 0
    with no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = VGG16_CIFAR10(inputs)
            test_correct += accuracy(outputs, targets)
            test_samples += targets.size(0)
    test_acc = test_correct / test_samples

    writer.add_scalars("VGG/Accuracy", {"Train": train_acc, "Test": test_acc}, global_step=epoch)
    writer.add_scalar("VGG/Loss", train_loss, global_step=epoch)
    print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")
