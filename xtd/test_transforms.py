import cv2
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

img = cv2.imread(r'xtd\hymenoptera\train\ants\512164029_c0a66b8498.jpg')

toTensor = transforms.ToTensor()
toNormalize = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
toCrop = transforms.RandomCrop(256)
toResize = transforms.Resize(128)
tran = transforms.Compose([toTensor, toNormalize, toCrop, toResize])

writer = SummaryWriter('logs')

for i in range(10):
    writer.add_image("composed", tran(img), i)

writer.flush()
writer.close()