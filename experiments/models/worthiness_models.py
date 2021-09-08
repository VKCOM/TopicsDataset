import torch
import torch.nn as nn
import torch.nn.functional as F

IMG_LEN = 1024
TXT_LEN = 300
N_WORTHINESSES = 2


class WorthinessNormModelBN(nn.Module):
    def __init__(self, d=128, drop=0.25):
        super().__init__()

        self.fc_img_1 = nn.Linear(IMG_LEN, d * 4)
        self.bn_img_1 = nn.BatchNorm1d(num_features=d * 4)
        self.fc_img_2 = nn.Linear(d * 4, d * 2)
        self.bn_img_2 = nn.BatchNorm1d(num_features=d * 2)

        self.fc_txt_1 = nn.Linear(TXT_LEN, d * 2)
        self.bn_txt_1 = nn.BatchNorm1d(num_features=d * 2)
        self.fc_txt_2 = nn.Linear(d * 2, d * 2)
        self.bn_txt_2 = nn.BatchNorm1d(num_features=d * 2)

        self.fc1 = nn.Linear(d * 4, d)
        self.bn1 = nn.BatchNorm1d(num_features=d)
        self.fc2 = nn.Linear(d, d)
        self.bn2 = nn.BatchNorm1d(num_features=d)
        self.out = nn.Linear(d, N_WORTHINESSES)

        self.dropout = nn.modules.Dropout(p=drop)

    def forward(self, inp_img, inp_txt):
        x_img = self.dropout(self.bn_img_1(F.relu(self.fc_img_1(inp_img))))
        x_img = self.dropout(self.bn_img_2(F.relu(self.fc_img_2(x_img))))

        x_txt = self.dropout(self.bn_txt_1(F.relu(self.fc_txt_1(inp_txt))))
        x_txt = self.dropout(self.bn_txt_2(F.relu(self.fc_txt_2(x_txt))))

        x = torch.cat((x_img, x_txt), 1)
        x = self.dropout(self.bn1(F.relu(self.fc1(x))))
        x = self.bn2(F.relu(self.fc2(x)))

        x = F.log_softmax(self.out(x), dim=1)
        return x
