import torch
import os
import torch.utils.data
import torchvision
import cv2
import numpy as np
import random

# 1.dataset
# 2.loss
# 3.train
#

class dataset(torch.utils.data.Dataset):
    def __init__(self, path, transform,TRAINED=True):
        self.path = path
        self.t = transform
        self.imgs = []
        self.labels = []
        self.num_samples = 0
        lines = open(self.path).readlines()
        print('read lines num:{}'.format(len(lines)))
        if TRAINED:
            lines=lines[:2600]
        else:
            lines=lines[2600:]
        i = 0
        for line in lines:
            # img_path:instances:x1:y1:x2:y2:clss
            arrs = line.strip().split()
            img_name = arrs[0]
            instances = int(arrs[1])
            img = cv2.imread(img_name)
            h, w, _ = img.shape  # h,w,c
            boxes = []
            for i in range(instances):
                box = np.zeros(5)
                box[0] = arrs[i * 5 + 6]
                box[1] = float(arrs[i * 5 + 2]) / w
                box[2] = float(arrs[i * 5 + 3]) / h
                box[3] = float(arrs[i * 5 + 4]) / w
                box[4] = float(arrs[i * 5 + 5]) / h
                boxes.append(box)
            self.imgs.append(img)
            self.labels.append(self.encode(torch.Tensor(boxes)))
        # i+=1
        # if i==1:
        #   break;
        self.num_samples = len(self.imgs)

    # print('imgs num:{},labels num:{}'.format(len(self.imgs),len(self.labels)))
    def encode(self, boxes):
        cell_size = 1 / 7.
        target = torch.zeros((7, 7, 30))
        for box in boxes:
            wh = box[3:] - box[1:3]
            cxcy = (box[3:] + box[1:3]) / 2
            ij = (cxcy / cell_size).ceil() - 1
            # print(ij)
            cxcy = (cxcy - ij * cell_size) / cell_size
            clss = box[0] + 10
            # print(int(ij[1]), np.minimum(int(ij[0]),6), 4)
            target[np.minimum(int(ij[1]),6), int(ij[0]), 4] = 1
            target[np.minimum(int(ij[1]),6), int(ij[0]), 9] = 1  # confidence
            # print(np.minimum(int(ij[1]),6), int(ij[0]))
            target[np.minimum(int(ij[1]),6), int(ij[0]), 0:2] = target[np.minimum(int(ij[1]),6), int(ij[0]), 5:7] = cxcy
            target[np.minimum(int(ij[1]),6), int(ij[0]), 2:4] = target[np.minimum(int(ij[1]),6), int(ij[0]), 7:9] = wh
            target[np.minimum(int(ij[1]),6), int(ij[0]), int(clss)] = 1
        return target

    def BGR2RGB(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def __getitem__(self, index):
        img = self.BGR2RGB(self.imgs[index])
        img = self.random_bright(img)
        #img = self.random_flip(img)
        #img = self.randomScale(img)
        img = self.randomBlur(img)
        img = self.RandomBrightness(img)
        img = self.RandomHue(img)
        img = self.RandomSaturation(img)
        #img = self.randomShift(img)
        
        target = self.labels[index]
        for t in self.t:
            img = cv2.resize(img, (224, 224))
            img = t(img)
        return img, target

    def __len__(self):
        return self.num_samples
    def BGR2HSV(self,img):
        return cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    def HSV2BGR(self,img):
        return cv2.cvtColor(img,cv2.COLOR_HSV2BGR)
    def RandomBrightness(self,bgr):#
        if random.random() < 0.5:
            hsv = self.BGR2HSV(bgr)
            h,s,v = cv2.split(hsv)
            adjust = random.choice([0.5,1.5])
            v = v*adjust
            v = np.clip(v, 0, 255).astype(hsv.dtype)
            hsv = cv2.merge((h,s,v))
            bgr = self.HSV2BGR(hsv)
        return bgr
    def RandomSaturation(self,bgr):
        if random.random() < 0.5:
            hsv = self.BGR2HSV(bgr)
            h,s,v = cv2.split(hsv)
            adjust = random.choice([0.5,1.5])
            s = s*adjust
            s = np.clip(s, 0, 255).astype(hsv.dtype)
            hsv = cv2.merge((h,s,v))
            bgr = self.HSV2BGR(hsv)
        return bgr
    def RandomHue(self,bgr):
        if random.random() < 0.5:
            hsv = self.BGR2HSV(bgr)
            h,s,v = cv2.split(hsv)
            adjust = random.choice([0.5,1.5])
            h = h*adjust
            h = np.clip(h, 0, 255).astype(hsv.dtype)
            hsv = cv2.merge((h,s,v))
            bgr = self.HSV2BGR(hsv)
        return bgr

    def randomBlur(self,bgr):
        if random.random()<0.5:
            bgr = cv2.blur(bgr,(5,5))
        return bgr

    def randomShift(self,bgr):
        #平移变换
        if random.random() <0.5:
            height,width,c = bgr.shape
            after_shfit_image = np.zeros((height,width,c),dtype=bgr.dtype)
            after_shfit_image[:,:,:] = (104,117,123) #bgr
            shift_x = random.uniform(-width*0.2,width*0.2)
            shift_y = random.uniform(-height*0.2,height*0.2)
            #print(bgr.shape,shift_x,shift_y)
            #原图像的平移
            if shift_x>=0 and shift_y>=0:
                after_shfit_image[int(shift_y):,int(shift_x):,:] = bgr[:height-int(shift_y),:width-int(shift_x),:]
            elif shift_x>=0 and shift_y<0:
                after_shfit_image[:height+int(shift_y),int(shift_x):,:] = bgr[-int(shift_y):,:width-int(shift_x),:]
            elif shift_x <0 and shift_y >=0:
                after_shfit_image[int(shift_y):,:width+int(shift_x),:] = bgr[:height-int(shift_y),-int(shift_x):,:]
            elif shift_x<0 and shift_y<0:
                after_shfit_image[:height+int(shift_y),:width+int(shift_x),:] = bgr[-int(shift_y):,-int(shift_x):,:]
            # print(after_shfit_image.shape)
            return after_shfit_image
        return bgr

    def randomScale(self,bgr):
        #固定住高度，以0.6-1.4伸缩宽度，做图像形变
        if random.random() < 0.5:
            scale = random.uniform(0.6,1.4)
            height,width,c = bgr.shape
            bgr = cv2.resize(bgr,(int(width*scale),height))
            return bgr
        return bgr

    def randomCrop(self,bgr):
        if random.random() < 0.9:
            height,width,c = bgr.shape
            h = random.uniform(0.6*height,height)
            w = random.uniform(0.6*width,width)
            x = random.uniform(0,width-w)
            y = random.uniform(0,height-h)
            x,y,h,w = int(x),int(y),int(h),int(w)
            img_croped = bgr[y:y+h,x:x+w,:]
            return img_croped
        return bgr

    def subMean(self,bgr,mean):
        mean = np.array(mean, dtype=np.float32)
        bgr = bgr - mean
        return bgr

    def random_flip(self,im):
        if random.random() < 0.5:
            im_lr = np.fliplr(im).copy()
            h,w,_ = im.shape
            return im_lr
        return im
    def random_bright(self, im, delta=16):
        alpha = random.random()
        if alpha > 0.3:
            im = im * alpha + random.randrange(-delta,delta)
            im = im.clip(min=0,max=255).astype(np.uint8)
        return im


def train():
    train_set = dataset('pedestrian.txt', [torchvision.transforms.ToTensor()])
    trains = torch.utils.data.DataLoader(dataset=train_set, batch_size=16, num_workers=12, shuffle=True)
    print('train data length:', len(trains))
    return trains


def test():
    test_set = dataset('pedestrian.txt', [torchvision.transforms.ToTensor()],TRAINED=False)
    tests = torch.utils.data.DataLoader(dataset=test_set, batch_size=16, num_workers=12, shuffle=True)
    print('test data length:', len(tests))
    return tests

if __name__=='__main__':
    data=train()
    # print(type(data))

    for i,(img,target) in enumerate(data):
        print(i,img.size(),target.size())
        break
