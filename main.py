import torch
import torch.nn as nn
import torch.nn.functional as F
from LCN import local_contrast_norm as lcn

nclasses = 43

class RSTNet(nn.Module):
    def __init__(self):
        super(RSTNet, self).__init__()
        self.lcn_radius = 7
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 200, 7, padding=2),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(200, 250, 4, padding=2),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(250, 350, 4, padding=2),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )
        self.fc1 = nn.Linear(350 * 6 * 6, 400)
        self.fc_drop = nn.Dropout(p=0.5)
        self.out = nn.Linear(400, nclasses)

    def forward(self, x):
        x = self.conv1(x)
        x = lcn(x, self.lcn_radius)  # ←
        x = self.conv2(x)
        x = lcn(x, self.lcn_radius)  # ←
        x = self.conv3(x)
        x = lcn(x, self.lcn_radius)  # ←
        x = x.view(-1, 350 * 6 * 6)
        x = F.leaky_relu(self.fc1(x))
        x = self.fc_drop(x)
        x = self.out(x)
        return x

def main():
    net = RSTNet()
    tmp = torch.randn(2,3,32,32)#出错了
    out = net(tmp)
    print(out.shape)

if __name__=='__main__':
    main()
