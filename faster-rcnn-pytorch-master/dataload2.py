import cv2
import numpy
import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset
import SimpleITK as sitk
import matplotlib.pyplot as plt


def get_random_data(annotation_line, jitter=.3, hue=.1, sat=1.5, val=1.5, random=True):
    '''r实时数据增强的随机预处理'''

    line = annotation_line.split()
    print(line)
    print(len(line))

    #nodules = np.array([np.array(list(map(int, nodules.split(',')))) for nodules in line[1:]])

    image = Image.open(line[0])
    iw, ih = image.size
    h, w = 600,600
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)
    dx = (w - nw) // 2
    dy = (h - nh) // 2

    image = image.resize((nw, nh), Image.BICUBIC)




    new_image = Image.new('RGB', (w, h), (128, 128, 128))
    new_image.paste(image, (dx, dy))
    image_data = np.array(new_image, np.float32)



   # radius = int((float(nodules[3])) / SP[0] / 2)

    plt.imshow(image, cmap='gray')
    plt.show()
    iw, ih = 512, 512
    h, w = 800, 800
    box = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]])
    print(box)


        # resize image
    print("111")
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)
    dx = (w - nw) // 2
    dy = (h - nh) // 2

    image = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new('RGB', (w, h), (128, 128, 128))
    new_image.paste(image, (dx, dy))
    image_data = np.array(new_image, np.float32)
    print("image_data:")
    print(image_data)

            # correct boxes
    box_data = np.zeros((len(box), 5))
    print(box_data)
    if len(box) > 0:

        np.random.shuffle(box)
        box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
        box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
        box[:, 0:2][box[:, 0:2] < 0] = 0
        box[:, 2][box[:, 2] > w] = w
        box[:, 3][box[:, 3] > h] = h
        box_w = box[:, 2] - box[:, 0]
        box_h = box[:, 3] - box[:, 1]
        box = box[np.logical_and(box_w > 1, box_h > 1)]
        box_data = np.zeros((len(box), 5))
        print(box)
        box_data[:len(box)] = box
        print(box_data)
        return image_data, box_data

"""
    # resize image
    new_ar = w / h * self.rand(1 - jitter, 1 + jitter) / self.rand(1 - jitter, 1 + jitter)
    scale = self.rand(.5, 1.5)
    if new_ar < 1:
        nh = int(scale * h)
        nw = int(nh * new_ar)
    else:
        nw = int(scale * w)
        nh = int(nw / new_ar)
    image = image.resize((nw, nh), Image.BICUBIC)

    # place image
    dx = int(self.rand(0, w - nw))
    dy = int(self.rand(0, h - nh))
    new_image = Image.new('RGB', (w, h), (128, 128, 128))
    new_image.paste(image, (dx, dy))
    image = new_image

    # flip image or not
    flip = self.rand() < .5
    if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)

    # distort image
    hue = self.rand(-hue, hue)
    sat = self.rand(1, sat) if self.rand() < .5 else 1 / self.rand(1, sat)
    val = self.rand(1, val) if self.rand() < .5 else 1 / self.rand(1, val)
    x = cv2.cvtColor(np.array(image, np.float32) / 255, cv2.COLOR_RGB2HSV)
    x[..., 0] += hue * 360
    x[..., 0][x[..., 0] > 1] -= 1
    x[..., 0][x[..., 0] < 0] += 1
    x[..., 1] *= sat
    x[..., 2] *= val
    x[x[:, :, 0] > 360, 0] = 360
    x[:, :, 1:][x[:, :, 1:] > 1] = 1
    x[x < 0] = 0
    image_data = cv2.cvtColor(x, cv2.COLOR_HSV2RGB) * 255  # numpy array, 0 to 1

    # correct boxes
    box_data = np.zeros((len(box), 5))
    if len(box) > 0:
        np.random.shuffle(box)
        box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
        box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
        if flip: box[:, [0, 2]] = w - box[:, [2, 0]]
        box[:, 0:2][box[:, 0:2] < 0] = 0
        box[:, 2][box[:, 2] > w] = w
        box[:, 3][box[:, 3] > h] = h
        box_w = box[:, 2] - box[:, 0]
        box_h = box[:, 3] - box[:, 1]
        box = box[np.logical_and(box_w > 1, box_h > 1)]  # discard invalid box
        box_data = np.zeros((len(box), 5))
        box_data[:len(box)] = box

    return image_data, box_data
"""

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


if __name__ == "__main__":
    annotation_path = 'E:/paper/back/rcnn/faster-rcnn-pytorch-master/faster-rcnn-pytorch-master/2007_train.txt'
    with open(annotation_path) as f:
        lines = f.readlines()
        img, y = get_random_data(lines[0], random=True)
        print(img,y)
