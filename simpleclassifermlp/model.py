import torch
import torchvision
class MLP(torch.nn.Module):
    def __init__(self,input_size,num_size):
        super(MLP,self).__init__()
        self.flatten=torch.nn.Flatten()
        self.model=torch.nn.Sequential(
            torch.nn.Linear(input_size,512),
            torch.nn.ReLU(),
            torch.nn.Linear(512,512),
            torch.nn.ReLU(),
            torch.nn.Linear(512,num_size),
            torch.nn.Softmax()
        )
    def forward(self, x):
        x=self.flatten(x)
        return self.model(x)

if __name__=='__main__':
    model=MLP(3*32*32,10)
    print(model(torch.randn(1,3,32,32).view(1,-1)).size())