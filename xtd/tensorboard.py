from torch.utils.tensorboard import SummaryWriter
import cv2

writer = SummaryWriter('logs')

img = cv2.imread(r'xtd\hymenoptera\train\ants\512164029_c0a66b8498.jpg')

writer.add_image("ant", img, 2, dataformats="HWC")

for i in range(100):
    writer.add_scalar("y = 3x", 3*i, i)

writer.flush()
writer.close()