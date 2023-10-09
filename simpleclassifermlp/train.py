import torch
import os
import torchvision
from model import MLP
import dataset
torch.cuda.manual_seed(111)

print('loading data...')
def train():
    train_set=dataset.CIFAR10()
    trains=torch.utils.data.DataLoader(dataset=train_set,batch_size=32,num_workers=4,shuffle=True)
    return trains

def test():
    test_set=dataset.CIFAR10(train=False)
    tests=torch.utils.data.DataLoader(dataset=test_set,batch_size=64,num_workers=4)
    return tests

data_train=train()
data_test=test()


model=MLP(3*32*32,10)
if torch.cuda.is_available():
    model.cuda()

optimizer=torch.optim.SGD(model.parameters(),lr=1e-3)
loss_function=torch.nn.CrossEntropyLoss()
path_model='./cifar10.pth'

if os.path.exists(path_model):
    state=torch.load(path_model)
    model.load_state_dict(state['model_state'])
    optimizer.load_state_dict(state['optimizer_state'])

max_accuracy=0.
try:
    epoch=-1
    while(True):
        epoch+=1
        loss_sum=0.0
        correct=0
        length=0
        model.train()
        for i,(img,target) in enumerate(data_train):
            if torch.cuda.is_available():
                img=img.cuda()
                target=target.cuda()
            model.zero_grad()
            pred=model(img)
            loss=loss_function(pred,target)
            loss.backward()
            optimizer.step()
            loss_sum+=loss.item()
        model.eval()
        for i,(img,target) in enumerate(data_test):
            if torch.cuda.is_available():
                img=img.cuda()
                target=target.cuda()
            pred=model(img)
            correct+=torch.sum(torch.max(pred, 1)[1]==target).item()
            length+=img.size()[0]
        print('epoch=',epoch,'train loss=',loss_sum/len(data_train),' validate accuracy=',correct/length)
        if max_accuracy<correct/length:
            torch.save({'model_state':model.state_dict(),'optimizer_state':optimizer.state_dict()},path_model)   
except KeyboardInterrupt:
    print('key interrupt...')
    