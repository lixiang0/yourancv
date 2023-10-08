import torch
import torchvision
class CNN2d(torch.nn.Module):
    def __init__(self,in_chanel,win,hin,num_class):
        super(CNN2d,self).__init__()
        self.in_chanel=in_chanel
        self.hin=hin
        self.win=win
        self.num_class=num_class
        self.conv1=torch.nn.Conv2d(self.in_chanel,16,3,stride=1,padding=1)
        self.norm=torch.nn.BatchNorm2d(16)
        self.relu=torch.nn.ReLU()
        temp=torch.randn(1,self.in_chanel,self.win,self.hin)
        temp=self.conv1(temp)
        self.linear=torch.nn.Linear(torch.numel(temp.data),self.num_class)
        self.softmax=torch.nn.Softmax()
    def forward(self, input):
        out=self.norm(self.conv1(input))
        out=self.relu(out)
        out=self.softmax(self.linear(out.view(len(input),-1)))
        return out

if __name__=='__main__':
    model=CNN2d(3,32,32,10)
    print(model(torch.randn(1,3,32,32)).size())