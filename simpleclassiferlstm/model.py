import torch
import torchvision
class RNN(torch.nn.Module):
    def __init__(self,input_size,num_size):
        super(RNN,self).__init__()

        self.lstm = torch.nn.LSTM(input_size, 1024, 2)
        self.linear=torch.nn.Sequential(
            torch.nn.Linear(1024,512),
            torch.nn.ReLU(),
            torch.nn.Linear(512,512),
            torch.nn.ReLU(),
            torch.nn.Linear(512,num_size),
            torch.nn.Softmax()
        )
    def forward(self, x):
        x, (hn, cn)=self.lstm(x)
        x=x[:,-1,:]
        x=self.linear(x)
        return x

if __name__=='__main__':
    rnn = RNN(3*32,10)
    input = torch.randn(10, 32,3*32 )
    output=rnn(input)
    print(output.size())