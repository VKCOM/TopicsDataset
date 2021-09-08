import torch
import torch.nn.functional as F
import torch.nn as nn


IMG_LEN = 1024
TXT_LEN = 300
N_CLASSES = 50


class AfterEncoderModel(nn.Module):
    def __init__(self, encoder, d=128, drop=0.5):
        super().__init__()

        self.encoder = encoder
        self.fc = nn.Linear(d, d)
        self.bn = nn.BatchNorm1d(num_features=d)
        self.out = nn.Linear(d, N_CLASSES)

        self.dropout = nn.modules.Dropout(p=drop)

    def forward(self, inp_img, inp_txt):
        x = self.encoder(inp_img, inp_txt)

        x = self.bn(F.relu(self.fc(x)))
        x = F.log_softmax(self.out(x), dim=1)
        return x


class AfterEncoderModelTrident(nn.Module):
    def __init__(self, encoder, d=128, drop=0.5):
        super().__init__()

        self.encoder = encoder

        self.fc1 = nn.Linear(d * 4, d)
        self.bn1 = nn.BatchNorm1d(num_features=d)
        self.fc2 = nn.Linear(d, d)
        self.bn2 = nn.BatchNorm1d(num_features=d)
        self.out = nn.Linear(d, N_CLASSES)

        self.out_img = nn.Linear(d * 2, N_CLASSES)
        self.out_txt = nn.Linear(d * 2, N_CLASSES)

        self.dropout = nn.modules.Dropout(p=drop)

    def forward(self, inp_img, inp_txt):
        x_img, x_txt = self.encoder(inp_img, inp_txt)
        x = torch.cat((x_img, x_txt), 1)
        x = self.dropout(self.bn1(F.relu(self.fc1(x))))
        x = self.bn2(F.relu(self.fc2(x)))
        x = F.log_softmax(self.out(x), dim=1)

        x_img = F.log_softmax(self.out_img(x_img), dim=1)
        x_txt = F.log_softmax(self.out_txt(x_txt), dim=1)

        return x, x_img, x_txt
