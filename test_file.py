import torch
from torch.autograd import Variable
import torch.nn as nn

#from net import vgg16
import torchvision.transforms as transforms
import cv2
import numpy as np   
import predict as pt
# print(torch.cuda.current_device())
# with torch.cuda.device(0):
#     print(torch.cuda.current_device());print('1')
#     print(torch.cuda.current_device())
#     import test_model
#     model=test_model.Model().cuda()
#     model=torch.nn.DataParallel(model).cuda()
#     model.load_state_dict(torch.load('yolo_cc.pth'))
#     import glob
#     imgs=glob.glob(r"./testimg/*.jpg")  
#     i=0
#     for img in imgs:
#         frame = cv2.imread(img)
#         i+=1
#         result = pt.predict_gpu_img(model,cv2.resize(frame,(448,448)))
#         for left_up,right_bottom,class_name,_,prob in result:
#             cv2.rectangle(frame,left_up,right_bottom,(0,255,0),2);
#             cv2.putText(frame,class_name,left_up,cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),1,cv2.LINE_AA)
#        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         #cv2.imshow('frame',frame)
#         cv2.imwrite('./testresult/{}.jpg'.format(str(i)),frame)
#         #if cv2.waitKey(1) & 0xFF == ord('q'):
#             #break

#     cv2.destroyAllWindows()
import glob
imgs=glob.glob(r'./testimg/*.JPG')+glob.glob(r'./testimg/*.jpg')


import model
model=model.YoloModel().cuda()
# model=torch.nn.DataParallel(model).cuda()
model.load_state_dict(torch.load('yolo.h5'))
for img in imgs:
    print(img)
    image = cv2.imread(img)
    image=cv2.resize(image,(224,224))
    result = pt.predict_gpu_img(model,image)
    for left_up,right_bottom,class_name,_,prob in result:
        if prob>0.7:
            cv2.rectangle(image,left_up,right_bottom,(0,255,0),2)
            cv2.putText(image,class_name,left_up,cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),1,cv2.LINE_AA)
            print(prob)
            cv2.imwrite('./testresult/{0}'.format(img.split('/')[-1]),image)
