import torch
import torchvision.transforms as transforms
import cv2
import numpy as np
from torchvision.ops import nms
VOC_CLASSES1 = (    # always index 0
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
'sheep', 'sofa', 'train', 'tvmonitor')
VOC_CLASSES=('background','person')
def decoder(pred):
    '''
    pred (tensor) 1x7x7x30
    return (tensor) box[[x1,y1,x2,y2]] label[...]
    '''
    boxes=[]
    cls_indexs=[]
    probs = []
    cell_size = 1./7
    pred = pred.data
    pred = pred.squeeze(0) #1x7x7x30 ==> 7x7x30

    contain1 = pred[:,:,4].unsqueeze(2)
    # print('contain1 size:{0}'.format(contain1.size()))
    contain2 = pred[:,:,9].unsqueeze(2)
    # print('contain2 size:{0}'.format(contain2.size()))
    contain = torch.cat((contain1,contain2),2)
    # print('contain2 size:{0}'.format(contain.size()))


    mask1 = contain > 0.9 #大于阈值
    mask2 = (contain==contain.max()) #we always select the best contain_prob what ever it>0.9
    mask = (mask1+mask2).gt(0)
    # print('mask size {0},mask2 size:{1}'.format(mask.size(),mask2.size()))
    # print(mask,mask2)
    min_score,min_index = torch.max(mask,2) #每个cell只选最大概率的那个预测框
    # print(min_score)
    for i in range(7):
        for j in range(7):
            if min_score[i,j] == 1:
                b=min_index[i,j]
                #print(i,j,b)
                #box size: x,y,w,h,c
                box = pred[i,j,b*5:b*5+4]
                contain_prob = torch.FloatTensor([pred[i,j,b*5+4]])
                xy = torch.FloatTensor([i,j])*cell_size #cell左上角  up left of cell
                box[:2] = box[:2]*cell_size + xy # return cxcy relative to image
                box_xy = torch.FloatTensor(box.size())#转换成xy形式    convert[cx,cy,w,h] to [x1,xy1,x2,y2]
                box_xy[:2] = box[:2] - 0.5*box[2:]
                box_xy[2:] = box[:2] + 0.5*box[2:]
                max_prob,cls_index = torch.max(pred[i,j,10:],0)
                boxes.append(box_xy.view(1,4))
                cls_indexs.append(cls_index.view(-1))
                probs.append(contain_prob)
    boxes = torch.cat(boxes,0) #(n,4)
    probs = torch.cat(probs,0) #(n,)
    
    cls_indexs = torch.cat(cls_indexs,0) #(n,)
    # print(cls_indexs)
    keep = nms(boxes,probs,.3)
    return boxes[keep],cls_indexs[keep],probs[keep]

def nms1(bboxes,scores,threshold=0.3):
    '''
    bboxes(tensor) [N,4]
    scores(tensor) [N,]
    '''
    x1 = bboxes[:,0]
    y1 = bboxes[:,1]
    x2 = bboxes[:,2]
    y2 = bboxes[:,3]
    areas = (x2-x1) * (y2-y1)

    _,order = scores.sort(0,descending=True)
    keep = []
    while order.numel() > 0:
        # print(order)
        i = order.view(-1).numpy()[0]
        keep.append(i)

        if order.numel() == 1:
            break

        xx1 = x1[order[1:]].clamp(min=x1[i])
        yy1 = y1[order[1:]].clamp(min=y1[i])
        xx2 = x2[order[1:]].clamp(max=x2[i])
        yy2 = y2[order[1:]].clamp(max=y2[i])

        w = (xx2-xx1).clamp(min=0)
        h = (yy2-yy1).clamp(min=0)
        inter = w*h

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        ids = (ovr<=threshold).nonzero().squeeze()
        if ids.numel() == 0:
            break
        order = order[ids+1]
    return torch.LongTensor(keep)
        
def predict_gpu_img(model,img):
    result = []
    h,w,_ = img.shape
    # img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = cv2.resize(img,(224,224))/255.
    
    # mean = (123,117,104)#RGB
    # img = img - np.array(mean,dtype=np.float32)
    
    #rgb image to tensor
    # print('before transform:{0}'.format(img.shape))
    transform = transforms.Compose([transforms.ToTensor(),])
    img = transform(img)
    # print('after transform:{0}'.format(img.size()))
    img = img[None,:,:,:]
    img = img.float().cuda()
    # print('',img.size())
    pred = model(img) #1x7x7x30
    pred = pred.cpu()
    boxes,cls_indexs,probs =  decoder(pred)

    for i,box in enumerate(boxes):
        x1 = int(box[0]*w)
        x2 = int(box[2]*w)
        y1 = int(box[1]*h)
        y2 = int(box[3]*h)
        cls_index = cls_indexs[i]
        cls_index = int(cls_index) # convert LongTensor to int
        prob = probs[i]
        prob = float(prob)
        result.append([(x1,y1),(x2,y2),VOC_CLASSES[cls_index],'image_name',prob])
    # print(len(result))
    return result

if __name__ == '__main__':
    # CUDA_VISIBLE_DEVICES=1 python predict.py
    import test_model
    model = test_model.YoloModel().cuda()
    model=torch.nn.DataParallel(model)
    model.load_state_dict(torch.load('yolo1.h5'))
    model.eval()
    import os
    # os.system('rm testresult/*.jpg')
    lines = open('pedestrian.txt','r').readlines()
    print('read lines num:{}'.format(len(lines)))
    SPLIT=int(.9*len(lines))
    for line in lines[SPLIT:]:
        arrs = line.strip().split()
        file_name=arrs[0].split('/')[-1][:-3]+'txt'
        writer=open(os.path.join('input/ground-truth/',file_name),'w')
        # img_path:instances:x1:y1:x2:y2:clss
        instances = int(arrs[1])
        for i in range(instances):
            writer.write(' '.join([str(item) for item in [arrs[i * 5 + 6],arrs[i * 5 + 2],arrs[i * 5 + 3],arrs[i * 5 + 4],arrs[i * 5 + 5]]]))# class_name, left, top, right, bottom = line.split()
            writer.write('\n')
    # for line in lines[SPLIT:]:
        # img=line.split()[0]
    # for img in imgs:
        # file_name=img.split('/')[-1][:-3]+'txt'
        writer=open(os.path.join('input/detection-results/',file_name),'w')
        image = cv2.imread(arrs[0])
        # image=cv2.resize(image,(224,224))
        result = predict_gpu_img(model,image)
        for left_up,right_bottom,class_name,_,prob in result:
            if prob>.3:
                print(left_up,right_bottom,class_name,prob)
                writer.write(' '.join([repr(item) for item in [VOC_CLASSES.index(class_name),prob,left_up[0],left_up[1],right_bottom[0],right_bottom[1]]]))# class_name, left, top, right, bottom = line.split()
                writer.write('\n')
                # cv2.rectangle(image,left_up,right_bottom,(0,255,0),2)
                # cv2.putText(image,class_name,left_up,cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),1,cv2.LINE_AA)
        writer.close()
        # cv2.imwrite('./testresult/{0}'.format(img.split('/')[-1]),image)




