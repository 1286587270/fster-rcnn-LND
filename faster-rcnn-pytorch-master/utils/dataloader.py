import cv2
import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset
import SimpleITK as sitk
from skimage.color import gray2rgb
import torchvision.transforms as transforms
from PIL import Image
class FRCNNDataset(Dataset):
    def __init__(self, train_lines, shape=[600,600], is_train=True,transform=None):

        self.train_lines = train_lines
        self.train_batches = len(train_lines)
        self.shape = shape
        self.is_train = is_train
        self.transform = transform
    def __len__(self):
        return self.train_batches

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a
        
    def get_random_data(self, annotation_line, jitter=.3, hue=.1, sat=1.5, val=1.5, random=True):
        '''r实时数据增强的随机预处理'''

        line = annotation_line.split()
        #print(line)
        if(len(line)>1):

            nodules = line[1].split(",")

            itkimage = sitk.ReadImage(line[0])
            numpyImage = sitk.GetArrayFromImage(itkimage)

            numpyOrigin = np.array(list(reversed(itkimage.GetOrigin())))
            numpySpacing = np.array(list(reversed(itkimage.GetSpacing())))
            OR = itkimage.GetOrigin()

            SP = itkimage.GetSpacing()
            #print(SP)
            nodules = np.array(nodules)
            #print(int(float(nodules[1])))
            image = sitk.GetArrayFromImage(itkimage)  # 获取数据，自动从同名的.raw文件读取

            x, y, z = int((int(float(nodules[0])) - int(OR[0])) / (SP[0])), int(
                (int(float(nodules[1])) - int(OR[1])) / (SP[1])), int((int(float(nodules[2])) - int(OR[2])) / (SP[2]))
            #print(x, y, z)
            image_data = image[z]
            radius = int((float(nodules[3])) / SP[0] / 2)
            #print(radius)
            #print(image)
            #image_data = gray2rgb(image_data)
            #print(image_data.size)
            #print(image_data.shape)
            image_data = np.expand_dims(image_data, 0)
            #print(image_data.shape)
            iw, ih = 512, 512
            h, w = 800, 800
            box = np.array([[np.array(x), np.array(y), np.array(x + 2 * radius), np.array(y + 2 * radius), 0]])
            #print(box)

            # resize image
            scale = min(w / iw, h / ih)
            nw = int(iw * scale)
            nh = int(ih * scale)
            dx = (w - nw) // 2
            dy = (h - nh) // 2

            #print("image_data:")
            #print(image_data)

            box_data = np.zeros((len(box), 5))
            #print(box_data)
            if len(box) > 0:
                box_data = np.zeros((len(box), 5))
                box_data[:len(box)] = box
                #print(box_data)

                return image_data, box_data
        else:
            itkimage = sitk.ReadImage(line[0])
            numpyImage = sitk.GetArrayFromImage(itkimage)




            # print(int(float(nodules[1])))
            image = sitk.GetArrayFromImage(itkimage)  # 获取数据，自动从同名的.raw文件读取

 
            image_data = image[32]
            image_data = np.expand_dims(image_data, 0)

            box = []
            box_data = np.zeros((len(box), 5))


            return image_data, box_data

    def __getitem__(self, index):
        img, y = self.get_random_data(self.train_lines[index], random=self.is_train)
        #img = np.transpose(img , [2,0,1])
        box = y[:, :4]
        label = y[:, -1]
        return img, box, label

# DataLoader中collate_fn使用
def frcnn_dataset_collate(batch):
    images = []
    bboxes = []
    labels = []
    for img, box, label in batch:
        images.append(img)
        bboxes.append(box)
        labels.append(label)
    images = np.array(images)
    return images, bboxes, labels
