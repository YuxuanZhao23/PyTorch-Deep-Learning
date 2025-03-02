from torch.utils.data import Dataset
import os
import cv2

class HymenoteraDataSet(Dataset):
    def __init__(self, path, label):
        self.path = path
        self.label = label
        self.imageListPath = os.listdir(os.path.join(path, label))

    def __getitem__(self, index):
        path = os.path.join(self.path, self.label, self.imageListPath[index])
        print(path)
        img = cv2.imread(path)
        self.showImage(img)
        return img
    
    def __len__(self):
        return len(self.imageListPath)
    
    def showImage(self, image):
        cv2.imshow(self.label, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
hda = HymenoteraDataSet(r'xtd\hymenoptera\train', 'ants')
hdb = HymenoteraDataSet(r'xtd\hymenoptera\train', 'bees')
hda.__getitem__(0)