import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from torchvision import datasets,transforms
from torch.utils.data import _utils
import torch.optim as optim

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


# hyper_parameters
EPOCH = 10
BATCH_SIZE = 1000
# accumulation_steps = 100
LR = 0.001#/accumulation_steps
DOWNLOAD = False


# get MNIST dataset
train_data = datasets.MNIST(
    root='./data',
    train=True,
    transform=transforms.ToTensor(),
    download=DOWNLOAD
)

train_loader = torch.utils.data.DataLoader(
    dataset=train_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

test_data = datasets.MNIST(
    root='./data',
    train=False,
)

test_x = Variable(torch.unsqueeze(test_data.test_data, dim=1)).type(torch.FloatTensor).cuda()/255
test_y = test_data.test_labels.cuda()


cnn = MyCnn()
print(cnn)
cnn.cuda()

optimizer = optim.Adam(cnn.parameters(),lr=LR)
loss_func = nn.CrossEntropyLoss()

for epoch in range(EPOCH):
    for step,(x,y) in enumerate(train_loader):
        torch.cuda.empty_cache()
        b_x = Variable(x).cuda()
        b_y = Variable(y).cuda()
        output = cnn(b_x)
        loss = loss_func(output,b_y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        # if (step+1)%accumulation_steps == 0:
        #     optimizer.step()
        #     optimizer.zero_grad()

        # test_output = cnn(test_x)
        # pred_y = torch.max(test_output,1)[1].cuda().data.squeeze()
        # accuracy = torch.sum(pred_y==test_y).type(torch.FloatTensor)/test_y.size(0)
        # print('Epoch: ',epoch,'|train loss: %.4f' % loss.item(),'|test accuracy: %.4f' % accuracy)

        if (step+1) % 50 == 0:
            test_output = cnn(test_x)
            pred_y = torch.max(test_output,1)[1].cuda().data.squeeze()
            accuracy = torch.sum(pred_y==test_y).type(torch.FloatTensor)/test_y.size(0)
            print('Epoch: ',epoch,'|train loss: %.4f' % loss.item(),'|test accuracy: %.4f' % accuracy)
torch.save(cnn.cpu(),'./model/MNIST_cnn.pkl')



