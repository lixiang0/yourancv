import torch
import torchvision
from model.vgg16 import decom_vgg16
from einops.layers.torch import Rearrange
#模型定义
class YoloModel(torch.nn.Module):
    def __init__(self):
        super(YoloModel, self).__init__()
        self.net,_ = decom_vgg16()
        self.head = torch.nn.Sequential(
            Rearrange('b c h w -> b (c h w)'),
            torch.nn.Linear(512*14*14,4096),# 7 * 7 * 30),  # 7*7*1024
            torch.nn.ReLU(),
            torch.nn.Dropout(),
            torch.nn.Linear(4096, 7 * 7 * 30),
        )
        self.model=torch.nn.Sequential(
            self.net,
            self.head,
            Rearrange('b (h w c) -> b h w c',h=7,w=7)
        )

    def forward(self, input):
        output = torch.sigmoid(self.model(input))
        return output



if __name__ == "__main__":
    #验证模型输入输出是否正确
    model = YoloModel().cuda()
    print(model)
    print(model(torch.randn(1,3,224,224).cuda()).size())