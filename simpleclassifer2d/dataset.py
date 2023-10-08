import torch
import torchvision
import torchvision.transforms as transforms
class CIFAR10(torch.utils.data.Dataset):
	def __init__(self,train=True):
		self.train=train
		self.trains,self.labels=self._init_trains(self.train)
		self.transforms = transforms.Compose(
				[transforms.ToTensor(),
				transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
	def _init_trains(self,train=True):
		dataloader=torchvision.datasets.CIFAR10('./data',train=train,download=True)
		x_train=dataloader.data  #类型是numpy
		y_train=dataloader.targets #类型是list
		return x_train,y_train
	def __getitem__(self,index):
		x=self.trains[index].astype('float32')#img float
		y=self.labels[index]
		x=self.transforms(x)#这里注意是hwc，经过转换后成为chw
		return x,y
	def __len__(self):
		return len(self.trains)
	
if __name__=='__main__':
	dataloader=torchvision.datasets.CIFAR10('./data',train=True,download=True)
	print('类别：',dataloader.classes)
	print('类别对应的ID：',dataloader.class_to_idx)
	x_train=torch.from_numpy(dataloader.data)
	y_train=dataloader.targets
	import cv2
	cv2.imwrite('1.jpg',x_train[0].numpy())

	print(x_train[0].size(),type(x_train[0]))
	print(y_train[0],type(y_train[0]))


	train_set=CIFAR10()
	trains=torch.utils.data.DataLoader(dataset=train_set,batch_size=8,num_workers=0,shuffle=True)
	for i,(img,target) in enumerate(trains):
		print(img.size(),target.size())
		break