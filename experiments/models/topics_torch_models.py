import torch
import torch.nn as nn
import torch.nn.functional as F


IMG_LEN = 1024
TXT_LEN = 300
N_CLASSES = 50


class NormModel(nn.Module):
    def __init__(self, drop=0.25, d=128):
        super().__init__()

        self.fc_img_1 = nn.Linear(IMG_LEN, 4 * d)
        self.fc_img_2 = nn.Linear(4 * d, 2 * d)

        self.fc_txt_1 = nn.Linear(TXT_LEN, 2 * d)
        self.fc_txt_2 = nn.Linear(2 * d, 2 * d)

        self.fc1 = nn.Linear(4 * d, d)
        self.fc2 = nn.Linear(d, d)
        self.out = nn.Linear(d, N_CLASSES)

        self.dropout = nn.modules.Dropout(p=drop)

    def forward(self, inp_img, inp_txt):
        x_img = F.relu(self.fc_img_1(inp_img))
        x_img = self.dropout(x_img)
        x_img = F.relu(self.fc_img_2(x_img))
        x_img = self.dropout(x_img)

        x_txt = F.relu(self.fc_txt_1(inp_txt))
        x_txt = self.dropout(x_txt)
        x_txt = F.relu(self.fc_txt_2(x_txt))
        x_txt = self.dropout(x_txt)

        x = torch.cat((x_img, x_txt), 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))

        x = F.log_softmax(self.out(x), dim=1)
        return x


class NormModelBN(nn.Module):
    def __init__(self, drop=0.5, d=128, n_classes=N_CLASSES):
        super().__init__()

        self.fc_img_1 = nn.Linear(IMG_LEN, 4 * d)
        self.bn_img_1 = nn.BatchNorm1d(num_features=4 * d)
        self.fc_img_2 = nn.Linear(4 * d, 2 * d)
        self.bn_img_2 = nn.BatchNorm1d(num_features=2 * d)

        self.fc_txt_1 = nn.Linear(TXT_LEN, 2 * d)
        self.bn_txt_1 = nn.BatchNorm1d(num_features=2 * d)
        self.fc_txt_2 = nn.Linear(2 * d, 2 * d)
        self.bn_txt_2 = nn.BatchNorm1d(num_features=2 * d)

        self.fc_1 = nn.Linear(4 * d, d)
        self.bn_1 = nn.BatchNorm1d(num_features=d)
        self.fc_2 = nn.Linear(d, d)
        self.bn_2 = nn.BatchNorm1d(num_features=d)
        self.out = nn.Linear(d, n_classes)

        self.dropout = nn.modules.Dropout(p=drop)

    def forward(self, inp_img, inp_txt):
        x_img = self.dropout(self.bn_img_1(F.relu(self.fc_img_1(inp_img))))
        x_img = self.dropout(self.bn_img_2(F.relu(self.fc_img_2(x_img))))

        x_txt = self.dropout(self.bn_txt_1(F.relu(self.fc_txt_1(inp_txt))))
        x_txt = self.dropout(self.bn_txt_2(F.relu(self.fc_txt_2(x_txt))))

        x = torch.cat((x_img, x_txt), 1)
        x = self.dropout(self.bn_1(F.relu(self.fc_1(x))))
        x = self.bn_2(F.relu(self.fc_2(x)))

        x = F.log_softmax(self.out(x), dim=1)
        return x


class NormModelTrident(nn.Module):
    def __init__(self, d=128, drop=0.25, residual=False):
        super().__init__()
        self.residual = residual

        self.fc_img_1 = nn.Linear(IMG_LEN, d * 4)
        self.fc_img_2 = nn.Linear(d * 4, d * 2)

        self.fc_txt_1 = nn.Linear(TXT_LEN, d * 2)
        self.fc_txt_2 = nn.Linear(d * 2, d * 2)

        self.fc1 = nn.Linear(d * 4, d if not residual else d * 2)
        self.fc2 = nn.Linear(d if not residual else d * 6, d)
        self.out = nn.Linear(d, N_CLASSES)

        self.out_img = nn.Linear(d * 2, N_CLASSES)
        self.out_txt = nn.Linear(d * 2, N_CLASSES)

        self.dropout = nn.modules.Dropout(p=drop)

    def forward(self, inp_img, inp_txt):
        x_img = F.relu(self.fc_img_1(inp_img))
        x_img = self.dropout(x_img)
        x_img = F.relu(self.fc_img_2(x_img))
        x_img = self.dropout(x_img)

        x_txt = F.relu(self.fc_txt_1(inp_txt))
        x_txt = self.dropout(x_txt)
        x_txt = F.relu(self.fc_txt_2(x_txt))
        x_txt = self.dropout(x_txt)

        x = torch.cat((x_img, x_txt), 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x if not self.residual else torch.cat((x_img, x_txt, x), 1)))

        out = F.log_softmax(self.out(x), dim=1)
        out_img = F.log_softmax(self.out_img(x_img), dim=1)
        out_txt = F.log_softmax(self.out_txt(x_txt), dim=1)

        return out, out_img, out_txt


class NormModelTridentBN(nn.Module):
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

        self.out = nn.Linear(d, N_CLASSES)
        self.out_img = nn.Linear(d * 2, N_CLASSES)
        self.out_txt = nn.Linear(d * 2, N_CLASSES)

        self.dropout = nn.modules.Dropout(p=drop)

    def forward(self, inp_img, inp_txt):
        x_img = self.dropout(self.bn_img_1(F.relu(self.fc_img_1(inp_img))))
        x_img = self.dropout(self.bn_img_2(F.relu(self.fc_img_2(x_img))))

        x_txt = self.dropout(self.bn_txt_1(F.relu(self.fc_txt_1(inp_txt))))
        x_txt = self.dropout(self.bn_txt_2(F.relu(self.fc_txt_2(x_txt))))

        x = torch.cat((x_img, x_txt), 1)
        x = self.dropout(self.bn1(F.relu(self.fc1(x))))
        x = self.bn2(F.relu(self.fc2(x)))

        out = F.log_softmax(self.out(x), dim=1)
        out_img = F.log_softmax(self.out_img(x_img), dim=1)
        out_txt = F.log_softmax(self.out_txt(x_txt), dim=1)

        return out, out_img, out_txt


class SelfAttentionModel1(nn.Module):
    def __init__(self):
        super().__init__()

        self.d = 256

        self.fc_img = nn.Linear(IMG_LEN, 128)
        self.fc_txt = nn.Linear(TXT_LEN, 128)

        self.fc_v = nn.Linear(self.d, self.d)
        self.fc_k = nn.Linear(self.d, self.d)
        self.fc_q = nn.Linear(self.d, self.d)

        self.fc_1 = nn.Linear(self.d, self.d)
        self.fc_2 = nn.Linear(self.d, self.d)
        self.out = nn.Linear(256, N_CLASSES)

        self.dropout = nn.modules.Dropout(p=0.25)

    def forward(self, inp_img, inp_txt):

        m = inp_img.shape[0]

        x_img = F.relu(self.fc_img(inp_img))
        x_img = self.dropout(x_img)

        x_txt = F.relu(self.fc_txt(inp_txt))
        x_txt = self.dropout(x_txt)

        x = torch.cat((x_img, x_txt), dim=1)

        v = self.fc_v(x)
        k = self.fc_k(x)
        q = self.fc_q(x)

        x_qk = torch.mm(q, torch.t(k)) / self.d ** (1./2)
        a = torch.nn.Softmax(dim=0)(torch.flatten(x_qk)).view(m, m)
        f = torch.mm(a, v)

        x = F.relu(self.fc_1(f))
        x = self.dropout(x)
        x = F.relu(self.fc_2(f))

        x = F.log_softmax(self.out(x), dim=1)
        return x


class GSAHelper(nn.Module):
    def __init__(self, d):
        super().__init__()

        self.d = d

        self.fc_k = nn.Linear(self.d, self.d)
        self.fc_q = nn.Linear(self.d, self.d)
        self.fc_kq = nn.Linear(self.d, self.d)

    def forward(self, k, q):
        m = k.shape[0]

        k_1 = self.fc_k(k)
        q_1 = self.fc_q(q)

        kq = nn.Sigmoid()(self.fc_kq(torch.mul(k_1, q_1)))

        k_2 = torch.mul(k, kq)
        q_2 = torch.mul(q, kq)

        mul = torch.mm(k_2, torch.t(q_2)) / self.d ** (1. / 2)
        a = nn.Softmax()(torch.flatten(mul)).view(m, m)
        return a


class GSA(nn.Module):
    def __init__(self, d):
        super().__init__()

        self.d = d

        self.fc_v = nn.Linear(self.d, self.d)
        self.fc_k = nn.Linear(self.d, self.d)
        self.fc_q = nn.Linear(self.d, self.d)
        self.gsa_helper = GSAHelper(self.d)

    def forward(self, x):
        m = x.shape[0]

        v = self.fc_v(x)
        k = self.fc_k(x)
        q = self.fc_q(x)

        a = self.gsa_helper(k, q)
        f = torch.mm(a, v)
        return f


class FFN(nn.Module):
    def __init__(self, d):
        super().__init__()

        self.fc_1 = nn.Linear(2 * d, 4 * d)
        self.drop = nn.Dropout(0.1)
        self.fc_2 = nn.Linear(4 * d, d)

    def forward(self, x_1, x_2):
        x = self.fc_1(torch.cat((x_1, x_2), 1))
        x = F.relu(x)
        x = self.drop(x)
        x = self.fc_2(x)
        return x


class UAModel1(nn.Module):
    def __init__(self, d=256):
        super().__init__()

        self.fc_img = nn.Linear(IMG_LEN, d // 2)
        self.fc_txt = nn.Linear(TXT_LEN, d // 2)

        self.d = d

        self.gsa_1 = GSA(d)
        self.ffn_1 = FFN(d)
        self.fc_out = nn.Linear(d, N_CLASSES)

    def forward(self, inp_img, inp_txt):
        x_img = self.fc_img(inp_img)
        x_txt = self.fc_txt(inp_txt)
        z = torch.cat((x_img, x_txt), 1)

        x = self.ffn_1(z, self.gsa_1(z))
        out = F.log_softmax(self.fc_out(x))
        return out


class UAModel2(nn.Module):
    def __init__(self, d=32):
        super().__init__()

        self.fc_img = nn.Linear(IMG_LEN, d // 2)
        self.fc_txt = nn.Linear(TXT_LEN, d // 2)

        self.d = d

        self.gsa_1 = GSA(d)
        self.ffn_1 = FFN(d)

        self.gsa_2 = GSA(d)
        self.ffn_2 = FFN(d)

        self.fc_out = nn.Linear(d, N_CLASSES)

    def forward(self, inp_img, inp_txt):
        x_img = self.fc_img(inp_img)
        x_txt = self.fc_txt(inp_txt)
        z = torch.cat((x_img, x_txt), 1)

        x = self.ffn_1(z, self.gsa_1(z))
        x = self.ffn_2(x, self.gsa_2(x))

        out = F.log_softmax(self.fc_out(x))
        return out


class TrivialModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(IMG_LEN + TXT_LEN, 64)
        self.dropout = nn.modules.Dropout(p=0.25)
        self.fc2 = nn.Linear(64, N_CLASSES)

    def forward(self, inp_img, inp_txt):
        x = torch.cat((inp_img, inp_txt), 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.log_softmax(self.fc2(x), dim=1)
        return x


class Encoder(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.fc_img = nn.Linear(IMG_LEN, d)
        self.fc_txt = nn.Linear(TXT_LEN, d)
        self.fc = nn.Linear(2 * d, 2 * d)

    def forward(self, inp_img, inp_txt):
        x_img = self.fc_img(inp_img)
        x_txt = self.fc_txt(inp_txt)

        x = torch.cat((x_img, x_txt), 1)
        x = F.relu(self.fc(x))
        return x


class Decoder(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.fc_img = nn.Linear(2 * d, IMG_LEN)
        self.fc_txt = nn.Linear(2 * d, TXT_LEN)

    def forward(self, x):
        x_img = self.fc_img(x)
        x_txt = self.fc_txt(x)
        return x_img, x_txt


class Autoencoder(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.encoder = Encoder(d)
        self.decoder = Decoder(d)

    def forward(self, inp_img, inp_txt):
        x = self.encoder(inp_img, inp_txt)
        x_img, x_txt = self.decoder(x)
        return x_img, x_txt


class TridentMTL(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = NormModelTridentBN(drop=0.5)
        self.sigma = nn.Parameter(torch.zeros(3))

    def forward(self, inp_img, inp_txt, target):
        outputs = self.model(inp_img, inp_txt)

        ls = [F.nll_loss(output, torch.argmax(target, dim=1)) for output in outputs]
        l = sum([(le / torch.exp(s)) for le, s in zip(ls, self.sigma)]) + sum([s / 2 for s in self.sigma])
        return l, ls
