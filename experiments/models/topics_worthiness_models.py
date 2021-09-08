import torch
import torch.nn as nn
import torch.nn.functional as F

IMG_LEN = 1024
TXT_LEN = 300
N_TOPICS = 50
N_WORTHINESSES = 2


class MultitargetTridentModelBN(nn.Module):
    def __init__(self, d=128, drop=0.25, worthiness_trident=False):
        super().__init__()
        self.worthiness_trident = worthiness_trident

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

        self.out_topics_common = nn.Linear(d, N_TOPICS)
        self.out_topics_img = nn.Linear(d * 2, N_TOPICS)
        self.out_topics_txt = nn.Linear(d * 2, N_TOPICS)

        self.out_worthiness_common = nn.Linear(d, N_WORTHINESSES)
        if self.worthiness_trident:
            self.out_worthiness_img = nn.Linear(d * 2, N_WORTHINESSES)
            self.out_worthiness_txt = nn.Linear(d * 2, N_WORTHINESSES)

        self.dropout = nn.modules.Dropout(p=drop)

    def forward(self, inp_img, inp_txt):
        x_img = self.bn_img_1(F.relu(self.fc_img_1(inp_img)))
        x_img = self.dropout(x_img)
        x_img = self.bn_img_2(F.relu(self.fc_img_2(x_img)))
        x_img = self.dropout(x_img)

        x_txt = self.bn_txt_1(F.relu(self.fc_txt_1(inp_txt)))
        x_txt = self.dropout(x_txt)
        x_txt = self.bn_txt_2(F.relu(self.fc_txt_2(x_txt)))
        x_txt = self.dropout(x_txt)

        x = torch.cat((x_img, x_txt), 1)
        x = self.dropout(self.bn1(F.relu(self.fc1(x))))
        x = self.bn2(F.relu(self.fc2(x)))

        out_topics_common = F.log_softmax(self.out_topics_common(x), dim=1)
        out_topics_img = F.log_softmax(self.out_topics_img(x_img), dim=1)
        out_topics_txt = F.log_softmax(self.out_topics_txt(x_txt), dim=1)

        out_worthiness_common = F.log_softmax(self.out_worthiness_common(x), dim=1)
        if self.worthiness_trident:
            out_worthiness_img = F.log_softmax(self.out_worthiness_img(x_img), dim=1)
            out_worthiness_txt = F.log_softmax(self.out_worthiness_txt(x_txt), dim=1)
            return (out_topics_common,
                    out_topics_img,
                    out_topics_txt,
                    out_worthiness_common,
                    out_worthiness_img,
                    out_worthiness_txt)
        else:
            return (out_topics_common,
                    out_topics_img,
                    out_topics_txt,
                    out_worthiness_common)
