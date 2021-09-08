import torch
import torch.nn as nn
import torch.nn.functional as F

N_CLASSES = 10
mnist_input_shape = (28, 28, 1)


class MnistModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_1 = nn.Conv2d(1, 32, 3, 1)
        self.conv_2 = nn.Conv2d(32, 64, 3, 1)

        self.dropout = nn.Dropout(p=0.5)
        self.fc_1 = nn.Linear(9216, 128)
        self.out = nn.Linear(128, N_CLASSES)

    def forward(self, x):
        x = F.relu(self.conv_1(x))
        x = F.relu(self.conv_2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = self.dropout(F.relu(self.fc_1(x)))
        out = F.log_softmax(self.out(x))
        return out
