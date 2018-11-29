import torch
import torchvision


class YoloModel(torch.nn.Module):
    def __init__(self):
        super(YoloModel, self).__init__()
        self.net = torchvision.models.vgg19_bn(pretrained=True)
        self.net.classifier = torch.nn.Sequential(
            torch.nn.Linear(25088, 4096),  # 7*7*1024
            torch.nn.ReLU(),
            torch.nn.Dropout(),
            torch.nn.Linear(4096, 7 * 7 * 30)
        )

    def forward(self, input):
        # for vgg
        N = input.size()[0]
        output = self.net(input)
        output = torch.nn.functional.sigmoid(output)
        return output.view(N, 7, 7, 30)



if __name__ == "__main__":
    model = YoloModel().cuda()
    print(model)
    print(model(torch.randn(1,3,224,224).cuda()).size())
    # import test_dataset

    # train_data = test_dataset.test()
    # for i, (img, target) in enumerate(train_data):
    #     print(img.shape)
    #     input = torch.autograd.Variable(img).cuda()
    #     print(model(input).size())
    #     break;
