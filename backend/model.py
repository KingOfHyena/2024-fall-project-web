import torch
import torch.nn as nn
import torch.nn.functional as F

class MyCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
            # [1, 26, 40]
            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),# [32, 26, 40]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        ) # [32, 13, 20]
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=96,
                kernel_size=3,
                padding=1
            ),# [64, 13, 20]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, ceil_mode=True)
        ) # [64, 7, 10]
        self.fc1 = nn.Linear(in_features=96*7*10, out_features=96)
        self.fc2 = nn.Linear(in_features=96, out_features=7)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output
