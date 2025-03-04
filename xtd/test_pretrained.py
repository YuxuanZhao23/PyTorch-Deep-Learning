import os
from torchvision.models import vgg16
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor, Compose, Resize
from torch.nn import Linear, CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch import no_grad, save, load
from torch.utils.tensorboard import SummaryWriter

def accuracy(outputs, targets):
    if outputs.dim() > 1 and outputs.size(1) > 1:
        outputs = outputs.argmax(dim=1)
    return (outputs == targets).sum().item()

batch_size = 64
num_epochs = 5
device = 'cuda'
checkpoint_path = 'vgg16_cifar10_last_layer.pth'
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
for param in VGG16_CIFAR10.classifier[-1].parameters():
    param.requires_grad = True

if os.path.exists(checkpoint_path):
    VGG16_CIFAR10.load_state_dict(load(checkpoint_path))
    print("Checkpoint loaded!")
else:
    print("No checkpoint found, training model from scratch.")

loss_function = CrossEntropyLoss()
optimizer = Adam(VGG16_CIFAR10.classifier[-1].parameters(), lr=1e-3)

VGG16_CIFAR10.to(device)

for epoch in range(num_epochs):
    VGG16_CIFAR10.train()
    train_loss, running_correct = 0, 0
    for i, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = VGG16_CIFAR10(inputs)
        batch_loss = loss_function(outputs, targets)
        batch_loss.backward()
        optimizer.step()
        
        train_loss += batch_loss.item()
        running_correct += (outputs.argmax(1) == targets).sum()

    train_acc = running_correct / len(train_dataset)

    VGG16_CIFAR10.eval() # 主要作用于 batch norm + dropout
    test_loss, test_correct = 0, 0
    with no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = VGG16_CIFAR10(inputs)
            test_loss += loss_function(outputs, targets).item()
            test_correct += (outputs.argmax(1) == targets).sum()
    test_acc = test_correct / len(test_dataset)

    writer.add_scalars("VGG/Accuracy", {"Train": train_acc, "Test": test_acc}, global_step=epoch)
    writer.add_scalars("VGG/Loss", {"Train": train_loss, "Test": test_loss}, global_step=epoch)
    print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")

save(VGG16_CIFAR10.state_dict(), checkpoint_path)
print(f"Model weights saved to {checkpoint_path}")
writer.close()