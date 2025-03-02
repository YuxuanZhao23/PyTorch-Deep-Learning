import cv2
from torchvision import transforms

img = cv2.imread(r'xtd\hymenoptera\train\ants\512164029_c0a66b8498.jpg')

tt = transforms.ToTensor()
t = tt(img)

print(t)