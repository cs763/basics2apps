import torch
import torch.nn as nn
import torch.nn.functional as F


def create_fcn(input_dim, output_dim):
    '''
        Create a Fully connected network by stacking multiple
        Linear/ReLU layers in the nn.Sequential container. The number
        of input and output neurons should be the same as the number
        of pixels in the image (108*108) and the number of classes (6),
        respectively
    '''
    model = nn.Sequential(
            nn.Linear(input_dim, 256),
            torch.nn.ReLU(),
            nn.Linear(256, 256),
            torch.nn.ReLU(),
            nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, output_dim),
        )

    # model = nn.Sequential(
    #         # Add your modules here
    #         nn.Linear(input_dim, output_dim),
    #         )

    return model

class Lenet(nn.Module):
    def __init__(self):
        super(Lenet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv3 = nn.Conv2d(16, 32, 5)
        self.fc1   = nn.Linear(32*10*10, 200)
        self.fc2   = nn.Linear(200, 44)
        self.fc3   = nn.Linear(44, 6)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv3(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out

