import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import imageio.v2 as img


class MyCnn(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.output_layer = nn.Linear(16*7*7,10)

    def forward(self,x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.reshape(x.size(0),-1)
        output = self.output_layer(x)
        return output


cnn = torch.load('model/MNIST_cnn.pkl')
path = input('请输入图片路径: ')
# path = './test.png'
image = img.imread(path)
image = image.mean(axis=2)/255
image = torch.tensor(image,dtype=torch.float).unsqueeze(0).unsqueeze(0)
# print(image.shape)
# image = transforms.ToTensor()(image)
result = cnn(image)
print(torch.max(result,1).indices[0].item())

# print(image)