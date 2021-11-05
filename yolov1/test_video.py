import torch
from torch.autograd import Variable
import torch.nn as nn

import torchvision.transforms as transforms
import cv2
import numpy as np   
import predict as pt
import time
import model
model=model.YoloModel().cuda()
model=torch.nn.DataParallel(model).cuda()
model.load_state_dict(torch.load('yolo.h5'))
# cap = cv2.VideoCapture('http://172.16.1.226:8081')
cap = cv2.VideoCapture('test.avi')
if not cap.isOpened():
	print('not open')

while(1):
	now=time.time()
	ret, origin = cap.read()
	if not ret:
		break
	origin=cv2.resize(origin,(1024,768))
	h,w,_=origin.shape
	frame=cv2.resize(origin,(224,224))
	result = pt.predict_gpu_img(model,frame)
	for left_up,right_bottom,class_name,_,prob in result:
		if prob>.6:
			x1=int(left_up[0]*w/224.)
			y1=int(left_up[1]*h/224.)
			x2=int(right_bottom[0]*w/224.)
			y2=int(right_bottom[1]*h/224.)
			cv2.rectangle(origin,(x1,y1),(x2,y2),(0,255,0),1)
			cv2.putText(origin,'person',(x1,y1),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),1,cv2.LINE_AA)
   # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	spend=time.time()-now
	cv2.putText(origin,'{0:.3}FPS'.format(1/spend),(20,20),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),1,cv2.LINE_AA)
	# frame=cv2.resize(origin,(1024,768))
	cv2.imshow('frame',origin)
	# cv2.waitKey(100)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()
# import glob
# imgs=glob.glob('./testimg/*.jpg')



# for img in imgs:
#     image = cv2.imread(img)
#     result = pt.predict_gpu(model,img)
#     for left_up,right_bottom,class_name,_,prob in result:
#         cv2.rectangle(image,left_up,right_bottom,(0,255,0),2)
#         cv2.putText(image,class_name,left_up,cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),1,cv2.LINE_AA)
#         print(prob)

#     cv2.imwrite('./testresult/{0}'.format(img.split('/')[-1]),image)
