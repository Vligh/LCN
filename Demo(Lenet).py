# 作者：张鑫
# 时间：2022/8/30 15:57
import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = F.conv2d(self,)