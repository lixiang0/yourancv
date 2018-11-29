import torch
from torch.autograd import Variable
import model
import dataset
import os

import loss
import tqdm
import cv2
import predict as pt

model = model.YoloModel().cuda()
model = torch.nn.DataParallel(model).cuda()

print('preparing data...')
train_data = dataset.train()
test_data = dataset.test()
print('done')
MODEL_PATH='yolo.h5'
if os.path.exists(MODEL_PATH):
    print('loading saved state...')
    model.load_state_dict(torch.load(MODEL_PATH))
    print('loading done')
print('start traing...')
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)
# optimizer = torch.optim.Adadelta(model.parameters())
# optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)
loss_fun = loss.yololoss()
bestloss = 1e10
e = 0
# for e in range(1200):
while True:
    model.train()
    train_loss = 0.0
    t = tqdm.tqdm(total=len(train_data))
    for i, (img, target) in enumerate(train_data):
        t.update(1)
        # print(i,img.size(),target.size())
        optimizer.zero_grad()
        pred = model(img.cuda())
        # print(i,img.size(),target.size(),pred.size())
        loss = loss_fun(pred, target.cuda())
        loss.backward()
        # print(loss.item())
        # vis.plot_train_val(loss_train=loss.item())
        optimizer.step()
        train_loss += loss.item()
        t.set_postfix(training_loss=train_loss / (i + 1))
    train_loss = train_loss / len(train_data)
    t.close()
    model.eval()
    test_loss = 0.0
    t = tqdm.tqdm(total=len(test_data))
    for i,(img,target) in enumerate(test_data):
      t.update(1)
      pred=model(img.cuda())
      loss=loss_fun(pred,target.cuda())
      test_loss+=loss.item()
      t.set_postfix(training_loss=test_loss / (i + 1))
    t.close()
    test_loss=test_loss/len(test_data)
    str_save = ''
    if bestloss > test_loss:
        bestloss = test_loss
        # import glob
        # imgs = glob.glob(r'./testimg/*.jpg')
        # for img in imgs:
        #     image = cv2.imread(img)
        #     image=cv2.resize(image,(190,190))
        #     result = pt.predict_gpu_img(model,image)
        #     for left_up,right_bottom,class_name,_,prob in result:
        #         if prob>0.7:
        #             cv2.rectangle(image,left_up,right_bottom,(0,255,0),2)
        #             cv2.putText(image,class_name,left_up,cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),1,cv2.LINE_AA)
        #             # print(prob)
        #             cv2.imwrite('./testresult/{0}'.format(img.split('/')[-1]),image)
        torch.save(model.state_dict(), MODEL_PATH)
    print('epoch={} train loss={} test loss={} best_loss={}'.format(e, train_loss, 0, bestloss))
    e += 1
