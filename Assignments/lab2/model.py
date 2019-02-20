import torch
import torch.nn as nn
from torch.autograd import Variable

class SegNet(nn.Module):
    def __init__(self):
        super(SegNet, self).__init__()
        self.relu = nn.ReLU(inplace = True)
        self.maxpool = nn.MaxPool2d(kernel_size = 2, stride = 2)

        self.conv1 = nn.Conv2d(3, 16, bias = True, kernel_size = 3, padding = 1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, bias = True, kernel_size = 3, padding = 1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, bias = True, kernel_size = 3, padding = 1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 32, bias = True, kernel_size = 3, padding = 1)
        self.bn4 = nn.BatchNorm2d(32)
        self.conv5 = nn.Conv2d(32, 21, bias = True, kernel_size = 3, padding = 1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)

        out = self.conv2(out)
        out = self.relu(out)
        out = self.bn2(out)

        out = self.conv3(out)
        out = self.relu(out)
        out = self.bn3(out)

        out = self.conv4(out)
        out = self.relu(out)
        out = self.bn4(out)

        out = self.conv5(out)

        return out



if __name__ == "__main__":
    network = SegNet()
    input = torch.rand(5,3,256,256)
    output = network(Variable(input))
    print(output.size())
